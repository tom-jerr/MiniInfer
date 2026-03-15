#include <atomic>
#include <optional>

template <typename T>
class LockFreeStack {
 private:
  struct Node {
    T value;
    Node* next;
    explicit Node(const T& v) : value(v), next(nullptr) {}
  };

  std::atomic<Node*> head_{nullptr};

 public:
  LockFreeStack() = default;
  LockFreeStack(const LockFreeStack&) = delete;
  LockFreeStack& operator=(const LockFreeStack&) = delete;

  ~LockFreeStack() {
    Node* cur = head_.load(std::memory_order_relaxed);
    while (cur) {
      Node* next = cur->next;
      delete cur;
      cur = next;
    }
  }

  void push(const T& value) {
    Node* new_node = new Node(value);
    new_node->next = head_.load(std::memory_order_relaxed);

    while (!head_.compare_exchange_weak(new_node->next, new_node, std::memory_order_release,
                                        std::memory_order_relaxed)) {}
  }

  std::optional<T> pop() {
    Node* old_head = head_.load(std::memory_order_acquire);

    while (old_head) {
      Node* next = old_head->next;
      if (head_.compare_exchange_weak(old_head, next, std::memory_order_acq_rel,
                                      std::memory_order_acquire)) {
        T value = old_head->value;
        delete old_head;  // 面试时要主动说明：这里真实生产里有回收风险
        return value;
      }
    }
    return std::nullopt;
  }

  bool empty() const { return head_.load(std::memory_order_acquire) == nullptr; }
};
