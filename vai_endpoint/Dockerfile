FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-cu124.2-4.ubuntu2204.py311
COPY entrypoint.sh entrypoint.sh
RUN chmod -R 775 entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
