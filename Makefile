# huncho packaging targets.
#
# Fedora COPR's `make srpm` SCM build method actually consumes `.copr/Makefile`
# in the repo root (see that file). The `srpm` target here is a thin local
# wrapper that delegates to it, dropping the source RPM in the checkout root.

NAME    := huncho
VERSION := 0.1.0
TARBALL := $(NAME)-$(VERSION).tar.gz

.PHONY: srpm tarball clean

# Produce a source RPM in the current directory (local convenience; COPR runs
# `.copr/Makefile` directly).
srpm:
	make -f .copr/Makefile srpm outdir=. spec=packaging/rpm/$(NAME).spec

# Produce just the upstream source tarball.
tarball:
	make -f .copr/Makefile tarball

clean:
	rm -f $(TARBALL) *.src.rpm
