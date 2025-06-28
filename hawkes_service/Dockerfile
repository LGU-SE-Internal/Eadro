FROM ubuntu:24.04
ENV TZ=Asia/Shanghai
# 添加代理支持
ARG http_proxy
ARG https_proxy
ARG no_proxy
ENV http_proxy=$http_proxy
ENV https_proxy=$https_proxy
ENV no_proxy=$no_proxy
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list.d/ubuntu.sources && \
#     sed -i 's@//.*security.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list.d/ubuntu.sources && \
#     sed -i 's@//ports.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list.d/ubuntu.sources
# RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
#     --mount=type=cache,target=/var/lib/apt,sharing=locked \
#     apt update && apt-get install -y \
#     fish git
ENV UV_LINK_MODE=copy
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
RUN uv python install 3.8
RUN uv venv
COPY requirements.txt .
RUN uv pip install -r requirements.txt
COPY hawkes_server.py .
EXPOSE 8080
CMD [".venv/bin/python", "hawkes_server.py"]
