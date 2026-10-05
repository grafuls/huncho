#
# Fedora COPR "make_srpm" entry point.
#
# COPR clones this repository and invokes, from the checkout root:
#     make -f .copr/Makefile srpm outdir="<dir>" spec="<spec_path>"
# The `srpm` target must drop the built source RPM into $(outdir).
#
# It runs with root privileges inside a mock chroot. That chroot does not
# install a full git, so the source tarball is produced with `tar` (falling
# back to `git ls-files` only when git is actually available). COPR leaves
# %dist empty for SRPM builds.
#

NAME    := huncho
VERSION := 0.1.0
TARBALL := $(NAME)-$(VERSION).tar.gz

# COPR passes these as lowercase command-line variables (`outdir`, `spec`);
# provide sane defaults so the same file works for a local `make srpm` too.
spec   ?= packaging/rpm/$(NAME).spec
outdir ?= .

# Scratch tree used only while assembling the SRPM (excluded from the tarball).
WORK := .copr-work

.PHONY: srpm tarball clean

srpm:
	# `%sysusers_create_package` and the `%systemd_*` helpers ship in
	# systemd-rpm-macros; install it if the buildroot lacks it.
	@[ -f /usr/lib/rpm/macros.d/macros.systemd ] \
		|| dnf -y install systemd-rpm-macros >/dev/null 2>&1 || true
	rm -rf $(WORK)
	mkdir -p $(WORK)/SOURCES $(WORK)/SPECS
	# Build the tarball straight into the rpmbuild sources dir.
	$(MAKE) -f .copr/Makefile tarball TARBALL_DEST=$(WORK)/SOURCES/$(TARBALL)
	cp packaging/rpm/$(NAME).env           $(WORK)/SOURCES/
	cp packaging/rpm/$(NAME).service       $(WORK)/SOURCES/
	cp packaging/rpm/$(NAME).1             $(WORK)/SOURCES/
	cp packaging/rpm/$(NAME)-sysusers.conf $(WORK)/SOURCES/
	# `huncho-cuda.spec` is self-contained (it bakes `with_cuda` for the whole
	# body), so no %include is needed at binary-build time. Just copy whichever
	# spec COPR asked for.
	cp $(spec)                             $(WORK)/SPECS/
	# Assemble the source RPM into $(outdir) so COPR can collect it.
	rpmbuild -bs \
		--define "_topdir $(CURDIR)/$(WORK)" \
		--define "_srcrpmdir $(outdir)" \
		$(WORK)/SPECS/$(notdir $(spec))
	rm -rf $(WORK)

# Produce a source tarball with a single `<name>-<version>/` top directory so
# `%setup -q` works. The archive is first written to a scratch dir outside the
# tree, then moved to its destination, so `tar` never sees its own output
# (otherwise it aborts with "file changed as we read it").
tarball:
	@set -e; \
	scratch="$$(mktemp -d)"; \
	trap 'rm -rf "$$scratch"' EXIT; \
	if command -v git >/dev/null 2>&1 && [ -d .git ]; then \
		git ls-files -z | tar --null --files-from=- \
			--transform="s|^|$(NAME)-$(VERSION)/|" \
			--create --gzip --file "$$scratch/$(TARBALL)"; \
	else \
		tar --exclude=$(WORK) --exclude=.git --exclude=target \
			--exclude=$(TARBALL) -C $(CURDIR) \
			--transform="s|^\./||" --transform="s|^|$(NAME)-$(VERSION)/|" \
			--create --gzip --file "$$scratch/$(TARBALL)" .; \
	fi; \
	dest="$(if $(TARBALL_DEST),$(TARBALL_DEST),$(CURDIR)/$(TARBALL))"; \
	mv "$$scratch/$(TARBALL)" "$$dest"

clean:
	rm -f $(TARBALL) *.src.rpm
