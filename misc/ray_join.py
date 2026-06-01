import sys
import time

from ray.job_submission import JobSubmissionClient
from ray.dashboard.modules.job.common import JobStatus

def main():
    client = JobSubmissionClient(sys.argv[1])
    errors = 0
    start_ts = time.time()
    while errors < 100:
        try:
            time.sleep(10)
            jobs = client.list_jobs()

            if jobs == None or len(jobs) == 0:
                print(f'[{time.time() - start_ts:.1f}s] wait job start')
                continue
            running = 0
            for job in jobs:
                if job.status == 'RUNNING' or job.status == 'PENDING':
                    running += 1
            if running == 0:
                print(f'[{time.time() - start_ts:.1f}s] {len(jobs)} jobs finished')
                break
            print(f'[{time.time() - start_ts:.1f}s] {running} job running')

        except Exception as e:
            print(e)
            errors += 1

if __name__ == '__main__':
    main()