from ml_build import pipelinerunner as pipe
from ml_build.logger import get_logger

log = get_logger('ML Pipeline')

def main():
    try:
        log.info("Starting the Machine learning flow")
        result = pipe.pipelinerunner()
        if result is None or result != "successfull":
            raise Exception("Machine learning pipeline failed")
        log.info("Machine learning flow completed")
    except Exception as e:
        log.error(f"The pipeline failed in the model run with error: {e}")
        raise

if __name__ == "__main__":
    main()