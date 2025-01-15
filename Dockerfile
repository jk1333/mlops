FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/huggingface-text-generation-inference-cu124.2-4.ubuntu2204.py311
COPY tgi_with_monitor.sh tgi_with_monitor.sh
RUN chmod -R 775 tgi_with_monitor.sh
COPY monitor.sh monitor.sh
RUN chmod -R 775 monitor.sh
ENTRYPOINT ["./tgi_with_monitor.sh"]