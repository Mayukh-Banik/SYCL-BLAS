FROM intel/oneapi-basekit:latest

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone https://github.com/Mayukh-Banik/syBlast.git
WORKDIR /app/syBlast
RUN git submodule update --init --recursive

RUN cmake --preset tests && cmake --build --preset tests

CMD ["build/test_Validation", "--gtest_brief=1"]
