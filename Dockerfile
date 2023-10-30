FROM python:3.10.13-bullseye

RUN apt update && apt install  bash wget zlib1g cmake clang curl git zip tar -y && \
    curl -s "https://get.sdkman.io" | bash
RUN bash -c "source $HOME/.sdkman/bin/sdkman-init.sh && yes | sdk install java 17.0.9-graal"\
    && export JAVA_HOME=graalvm/
RUN wget https://downloads.apache.org/maven/maven-3/3.9.5/binaries/apache-maven-3.9.5-bin.tar.gz\
    && tar -zxvf apache-maven-3.9.5-bin.tar.gz\
    && bash -c "mv apache-maven-3.9.5 maven"\
    && export M2_HOME="$(pwd)/maven"\
    && export M2=$M2_HOME/bin\
    && export PATH=$M2:$PATH\
    && git clone --recursive https://github.com/powsybl/pypowsybl.git\
    && cd pypowsybl\
    && git fetch\
    && git checkout move_connectable_prototype
#     && pip3 install --upgrade pip setuptools\
#     && pip3 install -r requirements.txt\
#     && pip3 install .


# RUN cd package\
#     && pip3 install -r requirements.txt
#
# COPY . ./package

# CMD cd pypowsybl/tests && ls -la && pip3 list && python3 -m pytest
CMD cd SDKMAN_DIR && ls -la
