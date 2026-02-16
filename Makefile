.PHONY: validate test

validate:
	./scripts/validate.sh

test: validate
