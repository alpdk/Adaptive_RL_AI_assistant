
import time
import torch

from multiprocessing import Lock, Condition, Event, Manager, Value


class ModelRequests:
    """
    This class provide methods for parallel threading request models for the outputs.

    Attributes:
        model (torch.nn.Module): The model to use.
        num_threads (int): amount of threads, that will be making requests
        inputs ({}): what data will be sent to the model
        outputs ({}): what data will be received from the model
        ready_flags ({}): events for model results collection
        lock (threading.Lock): a lock object to manage requests made
        condition (threading.Condition): a condition object to manage requests made
        request_id (int): id of the request
    """
    def __init__(self, model, num_threads):
        """
        Constructor

        Parameters:
            model (torch.nn.Module): The model to use
            num_threads (int): amount of threads, that will be making requests
        """
        self.model = model
        self.num_threads = num_threads

        self.manager = Manager()
        self.inputs = self.manager.dict()
        self.outputs = self.manager.dict()

        self.request_id = Value('i', 0)
        self.request_lock = Lock()

        self.lock = Lock()
        self.condition = Condition(self.lock)

        self.ready_flags = self.manager.dict()

    def trigger_fn(self):
        """
        Method fot checking model request condition

        Returns:
            res (bool): make the model request or not
        """
        return len(self.inputs) == self.num_threads and self.num_threads > 0

    def make_request(self, input_tensor):
        """
        Method for making request to the model with other threads

        Args:
            input_tensor (torch.tensor): tensor with input data

        Returns:
            res (torch.tensor): tensor with output from the model
        """
        with self.condition:
            self.request_lock.acquire()
            my_id = self.request_id.value
            # print(f"{my_id} Lock")
            self.request_id.value += 1
            self.request_lock.release()
            # my_id = self.request_id
            # # print(f"Current thread: {self.request_id % self.num_threads + 1}")
            # self.request_id += 1

            self.inputs[my_id] = input_tensor
            self.ready_flags[my_id] = self.manager.Event()

            # start = time.time()

            # print(f"{my_id} execution check")
            if self.trigger_fn():
                self._run_model()
            else:
                # Wait until our output is ready
                while my_id not in self.outputs:
                    self.condition.wait()
            # end = time.time()
            # print(f"{my_id} execution time: {end - start:.4f} seconds")

            self.ready_flags[my_id].wait()
            result = self.outputs.pop(my_id)
            # result = tuple(elem for elem in result)
            # result = tuple(out.unsqueeze(0) for out in result)
            self.ready_flags.pop(my_id)

            return result

    def delete_thread(self):
        """
        Method for decreasing amount of threads at the same time
        """
        with self.condition:
            self.num_threads -= 1

            if self.trigger_fn():
                self._run_model()

    def _run_model(self):
        """
        Method for requesting the model
        """
        input_ids = list(self.inputs.keys())
        batch = torch.cat([self.inputs[i] for i in input_ids], dim=0)

        batch_outputs = self.model(batch)
        if not isinstance(batch_outputs, tuple):
            batch_outputs = (batch_outputs,)

        start = time.time()
        # shapes = tuple(tensor.shape for tensor in batch_outputs)
        # print(f'Batch size: {shapes}')

        for idx, request_id in enumerate(input_ids):
            # self.outputs[request_id] = batch_outputs[idx]
            self.outputs[request_id] = tuple(out[idx].unsqueeze(0).detach() for out in batch_outputs)
            # print(self.outputs[request_id])
            # for out in batch_outputs:
            #     print(out[idx])
            self.ready_flags[request_id].set()

        # split_outputs = [list(out.detach().split(1, dim=0)) for out in batch_outputs]
        #
        # for idx, request_id in enumerate(input_ids):
        #     self.outputs[request_id] = tuple(split_outputs[i][idx].unsqueeze(0) for i in range(len(batch_outputs)))
        #     self.ready_flags[request_id].set()

        end = time.time()
        print(f"Elapsed time: {end - start:.4f} seconds")

        self.inputs.clear()
        self.condition.notify_all()
