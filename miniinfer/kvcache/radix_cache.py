from datetime import time
from functools import partial
import heapq
import logging

import torch
from kvcache.interface import IPrefixCache, ITokenAllocator
from typing import List, Tuple, Any, Optional
import time
import hashlib

logger = logging.getLogger(__name__)


def _key_match_page_size1(key0: List[int], key1: List[int]) -> int:
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i


def _key_match_paged(key0: List[int], key1: List[int], page_size: int) -> int:
    min_len = min(len(key0), len(key1))
    i = 0
    while i < min_len:
        if key0[i : i + page_size] != key1[i : i + page_size]:
            break
        i += page_size

    return i


def get_child_key(key: List[int], page_size: int = 1):
    if page_size == 1:
        plain_key = key[0]
    else:
        plain_key = tuple(key[:page_size])
    return plain_key


class TreeNode:
    def __init__(self):
        self.children = {}  # {token_id -> TreeNode}
        self.parent = None
        self.key = []  # token ids
        self.value = None  # KV indices (List[int])
        self.lock_ref = 0
        self.last_access_time = 0
        self.hash_value = None  # List of SHA256 hex strings

    @property
    def evicted(self):
        return self.value is None


class RadixCache(IPrefixCache):

    def __init__(self, token_allocator: ITokenAllocator, page_size: int = 1):
        self.root = TreeNode()
        self.page_size = page_size
        self.root.lock_ref = 1  # 根节点永不驱逐
        # keep legacy references used elsewhere
        self.root_node = self.root
        self.protected_size_ = 0
        self.token_allocator = token_allocator
        self.time_counter = 0
        self.evictable_size_ = 0
        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = get_child_key
        else:
            logger.debug("Using paged radix tree with page size %d", self.page_size)
            self.key_match_fn = partial(_key_match_paged, page_size=self.page_size)
            self.get_child_key_fn = partial(get_child_key, page_size=self.page_size)
        if self.token_allocator is not None:
            self.device = self.token_allocator.device
        else:
            self.device = torch.device("cpu")

    def match_prefix(self, key: List[int]) -> Tuple[torch.Tensor, "TreeNode"]:
        if len(key) == 0:
            return torch.empty((0,), dtype=torch.int64), self.root
        if self.page_size > 1:
            page_aligned_len = self._align_len(len(key))
            key = key[:page_aligned_len]
        node = self.root
        access_time = time.monotonic()
        node.last_access_time = access_time

        child_key = self.get_child_key_fn(key)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            child.last_access_time = access_time
            prefix_len = self.key_match_fn(child.key, key)
            if prefix_len < len(child.key):
                new_node = self._split_node(child.key, child, prefix_len)
                value.append(new_node.value)
                node = new_node
                break
            else:
                value.append(child.value)
                node = child
                key = key[prefix_len:]

                if len(key):
                    child_key = self.get_child_key_fn(key)
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)
        return value, node

    def insert(self, key: List[int], value=None):
        if value is None:
            value = torch.tensor(key, dtype=torch.int64)
        logger.debug(f"Inserting val of length {len(value)}")
        return self._insert_helper(self.root, key, value)

    def evict(self, num_tokens: int):
        leaves = self._collect_leaves()
        eviction_heap = [(node.last_access_time, node) for node in leaves]
        heapq.heapify(eviction_heap)

        num_evicted = 0
        while num_evicted < num_tokens and len(eviction_heap):
            _priority, x = heapq.heappop(eviction_heap)

            self.token_allocator.free(x.value)
            num_evicted += len(x.value)
            self._delete_leaf(x)

            if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
                new_priority = x.parent.last_access_time
                heapq.heappush(eviction_heap, (new_priority, x.parent))

    def inc_lock_ref(self, node: TreeNode):
        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.key)
                self.protected_size_ += len(node.key)
                delta -= len(node.key)
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode):
        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.key)
                self.protected_size_ -= len(node.key)
                delta += len(node.key)
            node.lock_ref -= 1
            if node.parent is None:
                assert (
                    node is self.root_node
                ), f"This request holds the node from another tree"
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        # protected size refers to the size of the cache that is locked
        return self.protected_size_

    def total_size(self):
        total_size = 0
        stack = [self.root]
        while stack:
            current_node = stack.pop()
            total_size += (
                len(current_node.value) if current_node.value is not None else 0
            )
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size

    def pretty_print(self):
        self._print_helper(self.root, 0)
        print(f"#tokens: {self.total_size()}")

    def _align_len(self, length: int) -> int:
        """向下取整到 page_size 的倍数"""
        return (length // self.page_size) * self.page_size

    def _split_node(self, key: List[int], child: TreeNode, split_len: int):
        # new_node -> child
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        return new_node

    def _insert_helper(self, node: TreeNode, key: List[int], value):
        access_time = time.monotonic()
        node.last_access_time = access_time
        if len(key) == 0:
            return 0
        child_key = self.get_child_key_fn(key)
        # logger.debug(f"Inserting key with child key {child_key}")
        total_prefix_len = 0

        while len(key) > 0 and child_key in node.children.keys():
            node = node.children[child_key]
            node.last_access_time = access_time
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_len += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)
                logger.debug(f"Descending to child key {child_key}")

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            node.children[child_key] = new_node
            self.evictable_size_ += len(key)
            logger.debug(f"Created new node with key {child_key}")

        return total_prefix_len

    def _collect_leaves(self):
        ret_list = []
        stack = list(self.root_node.children.values())

        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) == 0:
                if cur_node.lock_ref == 0:
                    ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())

        return ret_list

    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key[:10],
                f"r={current_node.lock_ref}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == self.get_child_key_fn(
                    child.key
                ), f"{key=}, {self.get_child_key_fn(child.key)=}"

    def _delete_leaf(self, node):
        key = self.get_child_key_fn(node.key)
        v = node.parent.children.pop(key, None)
        assert v == node, f"parent does not have child key, {key}"

        self.evictable_size_ -= len(node.key)


if __name__ == "__main__":
    tree = RadixCache(token_allocator=None, page_size=2)

    # Example token id sequences (as lists of ints)
    tree.insert([1, 2, 3])
    tree.insert([1, 2, 3])
    tree.insert([1, 2, 4, 5])
    tree.insert([1, 2, 4, 5, 6, 7])
    tree.insert([8, 9, 10, 11, 12])
    tree.pretty_print()

    print(tree.match_prefix([1, 2, 3, 13, 14]))
    print(tree.match_prefix([1, 2, 3]))
    print(tree.match_prefix([8, 9, 10, 11, 12, 13]))

    tree2 = RadixCache(token_allocator=None, page_size=1)
    tree2.insert([1, 2, 3])
    tree2.insert([1, 2, 4, 5])
    tree2.insert([1, 2, 4, 5, 6, 7])
    tree2.insert([8, 9, 10, 11, 12])
    # tree2.pretty_print()
    print(tree2.match_prefix([1, 2, 3, 13, 14]))
    print(tree2.match_prefix([1, 2, 3]))
    print(tree2.match_prefix([8, 9, 10, 11, 12, 13]))
