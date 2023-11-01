FROM python:3.10.13-bullseye

RUN apt update && apt install bash wget zlib1g cmake clang curl git zip tar -y
RUN wget https://download.oracle.com/graalvm/17/latest/graalvm-jdk-17_linux-x64_bin.tar.gz\
    &&  tar -xzf graalvm-jdk-17_linux-x64_bin.tar.gz

RUN export JAVA_HOME=/graalvm-jdk-17.0.9+11.1\
    && export PATH=/graalvm-jdk-17.0.9+11.1/bin:$PATH\
    && wget https://downloads.apache.org/maven/maven-3/3.9.5/binaries/apache-maven-3.9.5-bin.tar.gz\
    && tar -zxvf apache-maven-3.9.5-bin.tar.gz\
    && bash -c "mv apache-maven-3.9.5 maven"\
    && export M2_HOME="$(pwd)/maven"\
    && export M2=$M2_HOME/bin\
    && export PATH=$M2:$PATH\
    && git clone --recursive https://github.com/powsybl/pypowsybl.git\
    && cd pypowsybl\
    && git fetch\
    && git checkout move_connectable_prototype\
    && pip3 install --upgrade pip setuptools\
    && pip3 install -r requirements.txt\
    && pip3 install .

COPY . ./package

RUN cd package\
    && pip3 install -r requirements.txt

ENV PATH "$(pwd)/package":$PATH
ENV PYTHONPATH "$(pwd)/package"

CMD export PYTHONPATH="$(pwd)/package" && cd package && python3 src/ScriptAirgo.py
