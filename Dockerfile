FROM public.ecr.aws/lambda/python:3.10

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN pip install -r requirements.txt

# Copy function code
COPY lambda_heat_benchmark.py ${LAMBDA_TASK_ROOT}
COPY lambda_vpc_test.py ${LAMBDA_TASK_ROOT}
COPY lambda_db_diagnostic.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD [ "lambda_heat_benchmark.handler" ]
