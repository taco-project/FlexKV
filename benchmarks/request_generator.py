import asyncio
import random
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple, Any

import torch
from tqdm import tqdm


@dataclass
class KVRequest:
    request_id: int
    device_id: int
    tp_rank: int
    user_id: int
    turn_id: int
    request_type: str  # "get" or "put"
    timestamp: float
    token_ids: torch.Tensor
    token_mask: torch.Tensor
    slot_mapping: Optional[torch.Tensor] = None

    def _tensor_to_str(self, tensor: Optional[torch.Tensor]):
        if tensor is None:
            return "None"
        return (f"length={len(tensor)}: "
               f"[{tensor[0].item()}, {tensor[1].item()}, ... {tensor[-2].item()}, {tensor[-1].item()}], "
               f"dtype={tensor.dtype}")

    def __repr__(self):
        content = (f"request_id: \t{self.request_id}\n"
                   f"tp_rank: \t{self.tp_rank}\n"
                   f"device_id: \t{self.device_id}\n"
                   f"user_id: \t{self.user_id}\n"
                   f"turn_id: \t{self.turn_id}\n"
                   f"request_type: \t{self.request_type}\n"
                   f"timestamp: \t{self.timestamp}\n"
                   f"token_ids: \t{self._tensor_to_str(self.token_ids)}\n"
                   f"token_mask: \t{self._tensor_to_str(self.token_mask)}\n"
                   f"slot_mapping: \t{self._tensor_to_str(self.slot_mapping)}\n")
        return content

@dataclass
class UserRequest:
    user_id: int
    turn_id: int
    input: torch.Tensor
    output: torch.Tensor
    history: Optional[torch.Tensor] = None

def generate_random_multiturn(num_user_requests: int,
                              max_num_turns: int,
                              system_prompt_length: int,
                              max_input_length: int,
                              max_output_length: int,
                              num_turns_ratio: float = 0.8,
                              input_length_ratio: float = 0.8,
                              output_length_ratio: float = 0.8) -> List[UserRequest]:
    requests = []
    token_id_range = 10000
    system_prompt = torch.randint(0, token_id_range, (system_prompt_length,))
    for i in range(num_user_requests):
        user_requests = []
        num_turns = random.randint(int(num_turns_ratio * max_num_turns), max_num_turns)
        history = system_prompt.clone()
        for j in range(num_turns):
            input_length = random.randint(int(input_length_ratio * max_input_length), max_input_length)
            output_length = random.randint(int(output_length_ratio * max_output_length), max_output_length)
            input_tokens = torch.randint(0, token_id_range, (input_length,))
            output_tokens = torch.randint(0, token_id_range, (output_length,))
            request = UserRequest(
                user_id=i,
                turn_id=j,
                input=input_tokens,
                output=output_tokens,
                history=history
            )
            history = torch.cat([history, input_tokens, output_tokens], dim=0)
            user_requests.append(request)
        requests.append(user_requests)
    return requests

class RequestGenerator:
    supported_datasets = ["random"]
    def __init__(self,
                 dataset: str,
                 dataset_path: Optional[str] = None,
                 num_user_requests: int = 1000,
                 max_num_turns: int = 10,
                 system_prompt_length: int = 1024,
                 max_input_length: int = 10240,
                 max_output_length: int = 1024,
                 num_gpus: int = 1,
                 tp_size: int = 1,
                 max_parallel_per_device: int = 4,
                 request_rate: int = 1000,
                 approx_ttft: float = 0.001,
                 approx_tpot: float = 0.0001,
                 put_per_output_tokens: int = 1000,
                 random_seed: int = 42):
        random.seed(random_seed)
        torch.manual_seed(random_seed)

        assert dataset in self.supported_datasets, "Not supported dataset"
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.user_requests = []
        if dataset == "random":
            self.user_requests = generate_random_multiturn(num_user_requests,
                                                              max_num_turns,
                                                              system_prompt_length,
                                                              max_input_length,
                                                              max_output_length)

        self.num_gpus = num_gpus
        self.tp_size = tp_size
        assert self.num_gpus % self.tp_size == 0
        self.num_engines = self.num_gpus // self.tp_size

        self.request_rate = request_rate  # requests per second
        self.current_parallel = [0] * self.num_engines
        self.max_parallel_per_device = max_parallel_per_device

        self.ready_queues = [asyncio.Queue() for _ in range(self.num_engines)]
        self.kvrequest_queue = asyncio.Queue()
        self.request_counter = 0

        self.approx_ttft = approx_ttft
        self.approx_tpot = approx_tpot
        self.put_per_output_tokens = put_per_output_tokens

        self.total_requests = 0
        for user_request in self.user_requests:
            self.total_requests += len(user_request)
        print(f"total requests: {self.total_requests}")
        self.pbar = tqdm(total=self.total_requests)
        self.lock = asyncio.Lock()

    async def init_ready_queues(self):
        for i in range(0, len(self.user_requests)):
            user_request = self.user_requests[i][0]
            await self.ready_queues[i % self.num_engines].put(user_request)

    async def inc_request_id(self) -> int:
        async with self.lock:
            old_counter = self.request_counter
            self.request_counter += 1
            return old_counter

    async def inc_parallel(self, engine_id: int) -> None:
        async with self.lock:
            self.current_parallel[engine_id] += 1
            assert self.current_parallel[engine_id] > 0
            assert self.current_parallel[engine_id] <= self.max_parallel_per_device

    async def dec_parallel(self, engine_id: int) -> None:
        async with self.lock:
            self.current_parallel[engine_id] -= 1
            assert self.current_parallel[engine_id] >= 0
            assert self.current_parallel[engine_id] <= self.max_parallel_per_device

    async def get_parallel(self, engine_id: int) -> int:
        async with self.lock:
            return self.current_parallel[engine_id]

    async def simulate(self, engine_id: int, user_request: UserRequest) -> None:
        try:
            user_id = user_request.user_id
            turn_id = user_request.turn_id
            input = user_request.input
            output = user_request.output
            history = user_request.history
            sequence = input.clone() if history is None else torch.cat([history, input], dim=0)
            # get before prefill
            for tp_rank in range(self.tp_size):
                await self.kvrequest_queue.put(KVRequest(request_id=await self.inc_request_id(),
                                                     device_id=engine_id * self.tp_size + tp_rank,
                                                     tp_rank=tp_rank,
                                                     user_id=user_id,
                                                     turn_id=turn_id,
                                                     request_type="get",
                                                     timestamp=time.time(),
                                                     token_ids=sequence,
                                                     token_mask=torch.ones_like(sequence)))
            # prefill
            await asyncio.sleep(self.approx_ttft)
            # put after prefill
            for tp_rank in range(self.tp_size):
                await self.kvrequest_queue.put(KVRequest(request_id=await self.inc_request_id(),
                                                     device_id=engine_id * self.tp_size + tp_rank,
                                                     tp_rank=tp_rank,
                                                     user_id=user_id,
                                                     turn_id=turn_id,
                                                     request_type="put",
                                                     timestamp=time.time(),
                                                     token_ids=sequence,
                                                     token_mask=torch.ones_like(sequence)))
            # decode
            for i in range(0, len(output), self.put_per_output_tokens):
                await asyncio.sleep(self.approx_tpot * len(output[i:i+self.put_per_output_tokens]))
                sequence = torch.cat([sequence, output[i:i+self.put_per_output_tokens]], dim=0)
                for tp_rank in range(self.tp_size):
                    await self.kvrequest_queue.put(KVRequest(request_id=await self.inc_request_id(),
                                                         device_id=engine_id * self.tp_size + tp_rank,
                                                         tp_rank=tp_rank,
                                                         user_id=user_id,
                                                         turn_id=turn_id,
                                                         request_type="put",
                                                         timestamp=time.time(),
                                                         token_ids=sequence,
                                                         token_mask=torch.ones_like(sequence)))

        except Exception as e:
            print(f"Request failed: {e}")

        finally:
            self.pbar.update(1)
            await self.dec_parallel(engine_id)
            if turn_id < len(self.user_requests[user_id]) - 1:
                await self.ready_queues[engine_id].put(self.user_requests[user_id][turn_id + 1])

    def generate(self) -> List[KVRequest]:
        if self.pbar.n == self.pbar.total:
            return []
        async def request_loop():
            await self.init_ready_queues()
            while True:
                wait_new_request = False
                for i in range(self.num_engines):
                    if await self.get_parallel(i) < self.max_parallel_per_device:
                        try:
                            new_request = self.ready_queues[i].get_nowait()
                        except asyncio.QueueEmpty:
                            new_request = None
                        if new_request:
                            asyncio.create_task(self.simulate(i, new_request))
                            await self.inc_parallel(i)
                        wait_new_request = True
                if not wait_new_request:
                    await asyncio.sleep(0.05)
                    continue
                if self.pbar.n == self.pbar.total:
                    break
                sleep_time = random.normalvariate(1 / self.request_rate, 0.01)
                await asyncio.sleep(sleep_time)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(request_loop())
        loop.close()
        self.pbar.close()
        return [self.kvrequest_queue.get_nowait() for _ in range(self.kvrequest_queue.qsize())]

if __name__ == "__main__":
    request_generator = RequestGenerator(dataset="random",
                                         dataset_path=None,
                                         num_user_requests=3,
                                         max_num_turns=3,
                                         system_prompt_length=1024,
                                         max_input_length=10240,
                                         max_output_length=1024,
                                         num_gpus=4,
                                         tp_size=4,
                                         max_parallel_per_device=16,
                                         request_rate=1000,
                                         approx_ttft=0.001,
                                         approx_tpot=0.0001,
                                         put_per_output_tokens=1000)
    reqs: List[KVRequest] = request_generator.generate()
    print("-" * 80)
    print(type(reqs[0]))
    for req in reqs:
        print(req)
        print("-" * 80)
