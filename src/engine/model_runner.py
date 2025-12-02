


class ModelRunner:
  def call(self, method_name, *args):
      # if self.world_size > 1 and self.rank == 0:
      #     self.write_shm(method_name, *args)
      method = getattr(self, method_name, None)
      return method(*args)