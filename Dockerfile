FROM public.ecr.aws/lambda/python:3.10

RUN yum install -y gcc gcc-c++ make && yum clean all

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY RAG ./RAG

CMD ["RAG.app.handler"]
COPY artifacts /artifacts
