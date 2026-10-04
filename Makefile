# huncho packaging targets.
#
# Fedora COPR's SCM build method clones the repo and runs `make srpm` from the
# checkout root, then picks up the produced `.src.rpm`. The `srpm` target below
# is what COPR invokes; the others are local conveniences.

NAME    := huncho
VERSION := 0.1.0
TARBALL := $(NAME)-$(VERSION).tar.gz

.PHONY: srpm tarball clean

# Produce a source RPM in the current directory. Used by COPR.
srpm:
	./packaging/rpm/build-srpm.sh

# Produce just the upstream source tarball.
tarball:
	git ls-files -z \
		| tar --null --files-from=- \
			--transform="s|^|$(NAME)-$(VERSION)/|" \
			--create --gzip --file $(TARBALL)

clean:
	rm -f $(TARBALL) *.src.rpm
