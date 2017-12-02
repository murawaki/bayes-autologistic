# -*- mode: Makefile -*-
#
# usage make -f THIS_FILE OUTDIR=output_basedir
#
OUTDIR := data/default
OPT := --seed=0 --norm_sigma=10.0 --gamma_scale=1.0 --sample_param_weight=3 --init_vh=1.0 # noadjust
# OPT := --seed=0 --distance_weighting --norm_sigma=10.0 --gamma_scale=1.0 --sample_param_weight=3 --init_vh=1.0 # noadjust_weight
# OPT := --seed=0 --distance_thres=100000 --norm_sigma=10.0 --gamma_scale=1.0 --sample_param_weight=3 --init_vh=1.0 # noadjust100
# OPT := --seed=0 --distance_thres=500000 --norm_sigma=10.0 --gamma_scale=1.0 --sample_param_weight=3 --init_vh=1.0 # noadjust500
# OPT := --seed=0 --distance_thres=5000000 --norm_sigma=10.0 --gamma_scale=1.0 --sample_param_weight=3 --init_vh=1.0 # noadjust5000
# OPT := --seed=0 --norm_sigma=10.0 --gamma_scale=1.0 --sample_param_weight=3 --init_vh=1.0 --use_m # noadjust_m
# OPT := --seed=0 --norm_sigma=10.0 --gamma_scale=0.1 --sample_param_weight=3 --init_vh=1.0 # noadjust_s0.1
# OPT := --seed=0 --norm_sigma=10.0 --gamma_scale=0.01 --sample_param_weight=3 --init_vh=1.0 # noadjust_s0.01
# OPT := --seed=0 --distance_thres=5000000 --norm_sigma=10.0 --gamma_scale=1.0 --sample_param_weight=3 --init_vh=1.0 --use_m # noadjust_m5000
# OPT := --seed=0 --distance_thres=500000 --norm_sigma=10.0 --gamma_scale=1.0 --sample_param_weight=3 --init_vh=1.0 --use_m # noadjust_m500
# OPT := --seed=0 --distance_thres=100000 --norm_sigma=10.0 --gamma_scale=1.0 --sample_param_weight=3 --init_vh=1.0 --use_m # noadjust_m100
NICE := nice -19

CV := 10
CV_MAX := $(shell expr $(CV) - 1)
FEAT_NUM := 82 # HARD-CODED
FEAT_MAX := $(shell expr $(FEAT_NUM) - 1)

# gen_mv feat cvi
define gen_mvi
$(OUTDIR)/$(1).$(2).done :
	$(NICE) sh exec.sh $(OUTDIR) $(1) mvi $(2) "--cvn=$(2) $(OPT)"
MVI_$(1) += $(OUTDIR)/$(1).$(2).done
MVI_ALL += $(OUTDIR)/$(1).$(2).done
endef

define gen_param
$(OUTDIR)/$(1).done :
	$(NICE) sh exec.sh $(OUTDIR) $(1) param -1 "$(OPT)"
PARAM_ALL += $(OUTDIR)/$(1).done
endef

$(foreach i,$(shell seq 0 $(FEAT_MAX)), \
  $(foreach j,$(shell seq 0 $(CV_MAX)), \
    $(eval $(call gen_mvi,$(i),$(j)))))
$(foreach i,$(shell seq 0 $(FEAT_MAX)), \
  $(eval $(call gen_param,$(i))))

mvi : $(MVI_ALL)
param : $(PARAM_ALL)

all : mvi param
