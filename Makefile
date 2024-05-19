

docker-run:
	docker run --rm -it \
	--name my-blog \
	--volume="$(PWD):/srv/jekyll:Z" \
	-p 4000:4000 \
	jekyll/jekyll:4.2.0 \
	jekyll serve
