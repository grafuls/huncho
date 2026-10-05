# Thin wrapper that reuses huncho.spec with the CUDA build mode enabled.
#
# COPR's `make_srpm` build method copies this spec AND the base huncho.spec into
# the rpmbuild SPECS directory (see .copr/Makefile), so the include below
# resolves via the rpmbuild topdir. Building `huncho-cuda` defines `with_cuda`,
# which makes huncho.spec produce the standalone CUDA-capable `huncho-cuda`
# package.
%define with_cuda 1
%include %{_topdir}/SPECS/huncho.spec
