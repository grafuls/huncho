#
# Fedora COPR "make_srpm" entry point.
#
# COPR clones this repository and invokes, from the checkout root:
#     make -f .copr/Makefile srpm outdir="<dir>" spec="<spec_path>"
# The `srpm` target must drop the built source RPM into $(outdir).
#
# It runs with root privileges inside a mock chroot, so we may install
# packages or touch the network. COPR leaves %dist empty for SRPM builds.
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

srpm: tarball
	# `%sysusers_create_package` and the `%systemd_*` helpers ship in
	# systemd-rpm-macros; install it if the buildroot lacks it.
	@[ -f /usr/lib/rpm/macros.d/macros.systemd ] \
		|| dnf -y install systemd-rpm-macros >/dev/null 2>&1 || true
	rm -rf $(WORK)
	mkdir -p $(WORK)/SOURCES $(WORK)/SPECS
	cp $(TARBALL)                          $(WORK)/SOURCES/
	cp packaging/rpm/$(NAME).env           $(WORK)/SOURCES/
	cp packaging/rpm/$(NAME).service       $(WORK)/SOURCES/
	cp packaging/rpm/$(NAME).1             $(WORK)/SOURCES/
	cp packaging/rpm/$(NAME)-sysusers.conf $(WORK)/SOURCES/
	cp $(spec)                             $(WORK)/SPECS/
	# Assemble the source RPM into $(outdir) so COPR can collect it.
	rpmbuild -bs \
		--define "_topdir $(CURDIR)/$(WORK)" \
		--define "_srcrpmdir $(outdir)" \
		$(WORK)/SPECS/$(notdir $(spec))
	rm -rf $(WORK)

# Produce the upstream source tarball with a single `<name>-<version>/` top
# directory so `%setup -q` works. Prefer `git ls-files` (tracked files only);
# fall back to archiving the whole checkout if git is unavailable in the
# buildroot.
tarball:
	@if command -v git >/dev/null 2>&1 && [ -d .git ]; then \
		git ls-files -z | tar --null --files-from=- \
			--transform="s|^|$(NAME)-$(VERSION)/|" \
			--create --gzip --file $(TARBALL); \
	else \
		tar --exclude=$(WORK) --exclude=.git --exclude=target \
			--exclude=$(TARBALL) -C $(CURDIR) \
			--transform="s|^\./||" --transform="s|^|$(NAME)-$(VERSION)/|" \
			--create --gzip --file $(TARBALL) .; \
	fi

clean:
	rm -f $(TARBALL) *.src.rpm
