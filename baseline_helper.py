import sys
sys.path.append('scratch')
from api_request_parallel_processor import *

import os
import asyncio
import aiohttp
import time 
import json
from collections import defaultdict as ddict
import logging

max_requests_per_minute=60 * 0.5
max_tokens_per_minute=60000 * 0.5
token_encoding_name="cl100k_base"
max_attempts=5
logging_level=logging.INFO

def append_to_jsonl_chatgpt(response,pred_info,filename: str) -> None:
	"""Append a json payload to the end of a jsonl file."""
	prediction=response["choices"][0]["message"]["content"].replace('\n','')
	# print(prediction)
	json_string = json.dumps({"pred_info":pred_info,"prediction":prediction})
	with open(filename, "a") as f:
		f.write(json_string + "\n")


def append_to_jsonl_davinci(response,pred_info,filename: str) -> None:
	"""Append a json payload to the end of a jsonl file."""
	prediction=response["choices"][0]["text"].replace('\n','')
	# print(prediction)
	json_string = json.dumps({"pred_info":pred_info,"prediction":prediction})
	with open(filename, "a") as f:
		f.write(json_string + "\n")


import json
def append_to_jsonl(content,filename: str) -> None:
	"""Append a json payload to the end of a jsonl file."""
	with open(filename, "a") as f:
		json_string = json.dumps(content)
		f.write(json_string + "\n")
def load_gpt_pred_info(result_file):
	preds=[]
	with open(result_file) as IN:
		cnt=0
		for line in IN:
			json_info=json.loads(line)
			#print(json_info)
			pred_info=json_info['pred_info']
			prediction=json_info['prediction']
			preds.append((pred_info,prediction))
			cnt+=1
	return preds
		
@dataclass
class APIRequest_Chatgpt:
	"""Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

	task_id: int
	request_json: dict
	token_consumption: int
	attempts_left: int
	result = []
	pred_info:str
	total_line_num:str
	async def call_api(
		self,
		request_url: str,
		request_header: dict,
		retry_queue: asyncio.Queue,
		save_filepath: str,
		status_tracker: StatusTracker,
	):
		"""Calls the OpenAI API and saves results."""
		logging.info(f"Starting request #{self.task_id} of total {self.total_line_num} contents.")
		error = None
		try:
			async with aiohttp.ClientSession() as session:
				async with session.post(
					url=request_url, headers=request_header, json=self.request_json,  proxy="http://127.0.0.1:7890"
				) as response:
					response = await response.json()
			if "error" in response:
				logging.warning(
					f"Request {self.task_id} failed with error {response['error']}"
				)
				status_tracker.num_api_errors += 1
				error = response
				if "Rate limit" in response["error"].get("message", ""):
					status_tracker.time_of_last_rate_limit_error = time.time()
					status_tracker.num_rate_limit_errors += 1
					status_tracker.num_api_errors -= 1  # rate limit errors are counted separately
			#logging.info(f"Starting request #{self.task_id}")
			#logging.info(response)

		except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
			logging.warning(f"Request {self.task_id} failed with Exception {e}")
			status_tracker.num_other_errors += 1
			error = e

		if error:
			self.result.append(error)
			if self.attempts_left:
				retry_queue.put_nowait(self)
			else:
				logging.error(f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}")
				append_to_jsonl([self.request_json, self.result], save_filepath)
				status_tracker.num_tasks_in_progress -= 1
				status_tracker.num_tasks_failed += 1
		else:
			#logging.info(response)
			append_to_jsonl_chatgpt(response,self.pred_info, save_filepath)
			status_tracker.num_tasks_in_progress -= 1
			status_tracker.num_tasks_succeeded += 1
			logging.debug(f"Request {self.task_id} saved to {save_filepath}")


async def process_api_requests_from_file(
	requests_filepath: str,
	save_filepath: str,
	request_url: str,
	api_key: str,
	max_requests_per_minute: float,
	max_tokens_per_minute: float,
	token_encoding_name: str,
	max_attempts: int,
	logging_level: int,
):
	"""Processes API requests in parallel, throttling to stay under rate limits."""
	# constants
	print(1111111111)
	seconds_to_pause_after_rate_limit_error = 15
	seconds_to_sleep_each_loop = 0.001  # 1 ms limits max throughput to 1,000 requests per second

	# initialize logging
	logging.basicConfig(level=logging_level)
	logging.debug(f"Logging initialized at level {logging_level}")

	# infer API endpoint and construct request header
	api_endpoint = api_endpoint_from_url(request_url)
	request_header = {"Authorization": f"Bearer {api_key}"}

	# initialize trackers
	queue_of_requests_to_retry = asyncio.Queue()
	task_id_generator = task_id_generator_function()  # generates integer IDs of 1, 2, 3, ...
	status_tracker = StatusTracker()  # single instance to track a collection of variables
	next_request = None  # variable to hold the next request to call

	# initialize available capacity counts
	available_request_capacity = max_requests_per_minute
	available_token_capacity = max_tokens_per_minute
	last_update_time = time.time()

	# initialize flags
	file_not_finished = True  # after file is empty, we'll skip reading it
	logging.debug(f"Initialization complete.")
	
	# initialize file reading
	total_line_num= sum(1 for line in open(requests_filepath))
	with open(requests_filepath) as file:
		# `requests` will provide requests one at a time
		requests = file.__iter__()
		logging.debug(f"File opened. Entering main loop")
		
		while True:
			# get next request (if one is not already waiting for capacity)
			if next_request is None:
				if not queue_of_requests_to_retry.empty():
					next_request = queue_of_requests_to_retry.get_nowait()
					logging.debug(f"Retrying request {next_request.task_id}: {next_request}")
				elif file_not_finished:
					try:
						# get new request
						json_info = json.loads(next(requests))
						pred_info=json_info['pred_info']
						request_json={"model":json_info['model'],"messages":json_info["messages"],'temperature':json_info['temperature']}
						#logging.info(request_json)
						next_request = APIRequest_Chatgpt(
							task_id=next(task_id_generator),
							request_json=request_json,
							token_consumption=num_tokens_consumed_from_request(request_json, api_endpoint, token_encoding_name),
							attempts_left=max_attempts,pred_info=pred_info,total_line_num=str(total_line_num)
						)
						status_tracker.num_tasks_started += 1
						status_tracker.num_tasks_in_progress += 1
						logging.debug(f"Reading request {next_request.task_id}: {next_request}")
					except StopIteration:
						# if file runs out, set flag to stop reading it
						logging.debug("Read file exhausted")
						file_not_finished = False

			# update available capacity
			current_time = time.time()
			seconds_since_update = current_time - last_update_time
			available_request_capacity = min(
				available_request_capacity + max_requests_per_minute * seconds_since_update / 60.0,
				max_requests_per_minute,
			)
			available_token_capacity = min(
				available_token_capacity + max_tokens_per_minute * seconds_since_update / 60.0,
				max_tokens_per_minute,
			)
			last_update_time = current_time

			# if enough capacity available, call API
			if next_request:
				next_request_tokens = next_request.token_consumption
				if (
					available_request_capacity >= 1
					and available_token_capacity >= next_request_tokens
				):
					# update counters
					available_request_capacity -= 1
					available_token_capacity -= next_request_tokens
					next_request.attempts_left -= 1

					# call API
					asyncio.create_task(
						next_request.call_api(
							request_url=request_url,
							request_header=request_header,
							retry_queue=queue_of_requests_to_retry,
							save_filepath=save_filepath,
							status_tracker=status_tracker,
						)
					)
					next_request = None  # reset next_request to empty

			# if all tasks are finished, break
			if status_tracker.num_tasks_in_progress == 0:
				break

			# main loop sleeps briefly so concurrent tasks can run
			await asyncio.sleep(seconds_to_sleep_each_loop)

			# if a rate limit error was hit recently, pause to cool down
			seconds_since_rate_limit_error = (time.time() - status_tracker.time_of_last_rate_limit_error)
			if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
				remaining_seconds_to_pause = (seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error)
				await asyncio.sleep(remaining_seconds_to_pause)
				# ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
				logging.warn(f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}")

		# after finishing, log final status
		logging.info(f"""Parallel processing complete. Results saved to {save_filepath}""")
		if status_tracker.num_tasks_failed > 0:
			logging.warning(f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}.")
		if status_tracker.num_rate_limit_errors > 0:
			logging.warning(f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate.")


'''
### this function will generate a json file that is for gpt's multithreading process
### INPUT: pred_info, contents, output_request_path, iteration
	
### OUTPUT: output_request_file


'''
def generate_message_json(pred_infos, contents,temperature, output_request_path='temp.json',iterations=1):
	jobs=[]
	for pred_info, content,t in zip(pred_infos,contents,temperature):
		for i in range(iterations):
			jobs.append({"pred_info":pred_info,"model": "gpt-3.5-turbo-0301","messages":[{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": content}],'temperature':t,'max_tokens':3})
		  
	with open(output_request_path, "w+") as f:
		for job in jobs:
			json_string = json.dumps(job)
			f.write(json_string + "\n")
	print(f"Successfully Write to {output_request_path}")

def generate_message_json_davinci(pred_infos, contents,temperature, output_request_path='temp.json',iterations=1):
	jobs=[]
	for pred_info, content,t in zip(pred_infos,contents,temperature):
		for i in range(iterations):
			jobs.append({"pred_info":pred_info,"model": "text-davinci-003","prompt":content,"temperature":t})
		  
	with open(output_request_path, "w+") as f:
		for job in jobs:
			json_string = json.dumps(job)
			f.write(json_string + "\n")
	print(f"Successfully Write to {output_request_path}")



def run_multithread_gpt(request_file,output_chatgpt_predict_file):
	request_url="https://api.openai.com/v1/chat/completions"
	api_key="api key"
	max_requests_per_minute=60 * 0.5
	max_tokens_per_minute=60000 * 0.5
	token_encoding_name="cl100k_base"
	max_attempts=5
	logging_level=logging.INFO

	asyncio.run(
			process_api_requests_from_file(
				requests_filepath=request_file,
				save_filepath=output_chatgpt_predict_file,
				request_url=request_url,
				api_key=api_key,
				max_requests_per_minute=max_requests_per_minute,
				max_tokens_per_minute=max_tokens_per_minute,
				token_encoding_name=token_encoding_name,
				max_attempts=max_attempts,
				logging_level=logging_level,
			)
		)
	