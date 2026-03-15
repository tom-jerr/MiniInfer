#include <atomic>
#include <optional>

template <typename T>
class LockFreeQueue {
 private:
  struct Node {
    std::optional<T> value;
    std::atomic<Node*> next;

    Node() : value(std::nullopt), next(nullptr) {}
    explicit Node(const T& v) : value(v), next(nullptr) {}
  };

  std::atomic<Node*> head_;
  std::atomic<Node*> tail_;

 public:
  LockFreeQueue() {
    Node* dummy = new Node();
    head_.store(dummy, std::memory_order_relaxed);
    tail_.store(dummy, std::memory_order_relaxed);
  }

  LockFreeQueue(const LockFreeQueue&) = delete;
  LockFreeQueue& operator=(const LockFreeQueue&) = delete;

  ~LockFreeQueue() {
    Node* cur = head_.load(std::memory_order_relaxed);
    while (cur) {
      Node* next = cur->next.load(std::memory_order_relaxed);
      delete cur;
      cur = next;
    }
  }

  void enqueue(const T& value) {
    Node* new_node = new Node(value);

    while (true) {
      Node* tail = tail_.load(std::memory_order_acquire);
      Node* next = tail->next.load(std::memory_order_acquire);

      if (tail == tail_.load(std::memory_order_acquire)) {
        if (next == nullptr) {
          if (tail->next.compare_exchange_weak(next, new_node, std::memory_order_release,
                                               std::memory_order_relaxed)) {
            tail_.compare_exchange_weak(tail, new_node, std::memory_order_release,
                                        std::memory_order_relaxed);
            return;
          }
        } else {
          // 只是尝试推进 tail
          tail_.compare_exchange_weak(tail, next, std::memory_order_release,
                                      std::memory_order_relaxed);
        }
      }
    }
  }

  std::optional<T> dequeue() {
    while (true) {
      Node* head = head_.load(std::memory_order_acquire);
      Node* tail = tail_.load(std::memory_order_acquire);
      Node* next = head->next.load(std::memory_order_acquire);

      if (head == head_.load(std::memory_order_acquire)) {
        if (next == nullptr) {
          return std::nullopt;
        }
        // 只是 tail_ 还没来得及前移，此时不是空队列，继续推进 tail 就行了
        if (head == tail) {
          tail_.compare_exchange_weak(tail, next, std::memory_order_release,
                                      std::memory_order_relaxed);
          continue;
        }

        T value = *(next->value);
        if (head_.compare_exchange_weak(head, next, std::memory_order_acq_rel,
                                        std::memory_order_acquire)) {
          delete head;  // 真实生产环境仍需安全回收方案
          return value;
        }
      }
    }
  }
};
