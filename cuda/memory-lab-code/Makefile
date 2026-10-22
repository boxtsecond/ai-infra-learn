NVCC      ?= /usr/local/cuda/bin/nvcc
ARCH      ?= sm_90
NVCCFLAGS  = -O3 -arch=$(ARCH) -lineinfo

SRCDIR = src
BINDIR = bin

TARGETS = $(BINDIR)/devinfo \
          $(BINDIR)/l0_copy $(BINDIR)/l0_copy_vec \
          $(BINDIR)/l1_stride $(BINDIR)/l1_diag \
          $(BINDIR)/l2_offset $(BINDIR)/l3_layout \
          $(BINDIR)/l4_transpose $(BINDIR)/l5_bank

all: $(TARGETS)

$(BINDIR)/%: $(SRCDIR)/%.cu | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(BINDIR):
	mkdir -p $(BINDIR)

clean:
	rm -rf $(BINDIR)

.PHONY: all clean
