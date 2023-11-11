conda env export --file environment.yml
sed -i '/prefix.*/d' environment.yml
