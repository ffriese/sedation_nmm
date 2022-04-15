# credits to "Willian Paixao" <willian@ufpa.br>
#
FROM ubuntu:latest

LABEL version="1.1.0"

ENV DEBIAN_FRONTEND noninteractive

# Install all TeX and LaTeX dependences
RUN apt-get update && \
  apt-get install --yes --no-install-recommends \
  texlive-full  && \
  apt-get autoclean && apt-get --purge --yes autoremove && \
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


RUN apt-get update && \
  apt-get install --yes --no-install-recommends \
  python3 \
  python3-pip \
  git \
    ssh
#  && apt-get autoclean && apt-get --purge --yes autoremove && \
#  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip3 install Pygments PyRates matplotlib

RUN mkdir /data
WORKDIR /data
ADD docker/commands.sh /data/commands.sh
# ENV GIT_REPO=git@github.com:ffriese/sedation_nmm.git
ENTRYPOINT ["/bin/bash", "commands.sh"]


#### docker run --rm --mount type=bind,source="$(pwd)"/out_pdf,target=/out --mount type=bind,source="$(pwd)",target=/texdata -i xelatex
