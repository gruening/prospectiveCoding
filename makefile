# SIMOPTS = \
# -U POSTSPIKING \ # if defined use (smoothed backpropaged) output spike train instead of \phi(U)
# -U ETA \ # default in helper.c = 0.5
# -U FILENAME_POST \ # to record post activities
# -U FILENAME_PRE \ # to record pre activities if desired.
# -U CURRENTPRED \ # xxx what exactly does this switch on:
# -U FROZEN_POISSON \ # if neither FROZEN_POSSION or RAMPUPRATE are
# -U RAMPUPRATE \ # defined, as a default we have the linear with
# # time activation of indivudual input synapses  
# -U INPUT_AT_STIM \ # injected somatic current for stimulus.
# -U RECORD_PREACT \ # record also pre activities 
# -U ROTATIONS \ # drive with time varying stimulus from which
# # frozen poisson patterns are generated
# -U TAU_ALPHA \ # sets the macroscopic "rampUp" time constant

LIBS = -lgsl -lgslcblas -lm
#FLAGS = -std=c99 -O3
FLAGS = -std=c99 -O0 -g

# simulation. Why ETA bigger by factor 100 than default?
rampUp:
	gcc  $(FLAGS) -D ETA=50 -D FILENAME_POST=\"data/rampUpPost.dat\" -D FILENAME_PRE=\"data/rampUpPre.dat\" -c src/c/rampUp.c -o rampUp.o
	gcc  rampUp.o $(LIBS) -o bin/rampUp
	rm rampUp.o
#	time ./bin/rampUp

# simulation with CURRENTPRED on -- does it do anything else but switch on the inhibitory ones?
rampUpC:
	gcc  $(FLAGS) -D ETA=50 -D FILENAME_POST=\"data/rampUpPostC.dat\" -D FILENAME_PRE=\"data/rampUpPreC.dat\" -D CURRENTPRED  -c src/c/rampUp.c -o rampUpC.o
	gcc  rampUpC.o $(LIBS) -o bin/rampUpC
	rm rampUpC.o
#	time ./bin/rampUpC

# with fixed poisson
rampUpFP:
	gcc  $(FLAGS) -D FROZEN_POISSON -D  RECORD_PREACT -D FILENAME_POST=\"data/rampUpFPPost.dat\" -D FILENAME_PRE=\"data/rampUpFPPre.dat\" -c src/c/rampUp.c -o rampUpFP.o
	gcc  rampUpFP.o $(LIBS) -o bin/rampUpFP
	rm rampUpFP.o
#	time ./bin/rampUpFP

# with fixed poisson and currentPRED on
rampUpFPC:
	gcc  $(FLAGS) -D FROZEN_POISSON -D FILENAME_POST=\"data/rampUpFPPostC.dat\" -D FILENAME_PRE=\"data/rampUpFPPreC.dat\" -D CURRENTPRED  -c src/c/rampUp.c -o rampUpFPC.o
	gcc  rampUpFPC.o $(LIBS) -o bin/rampUpFPC
	rm rampUpFPC.o
#	time ./bin/rampUpFPC

# why different input_at_stim? With three sections of different rates
# and spikes generate randomly from it.
rampUpRate:
	gcc  $(FLAGS) -D RECORD_PREACT -D FILENAME_POST=\"data/rampUpRatePost.dat\" -D FILENAME_PRE=\"data/rampUpRatePre.dat\" -D INPUT_AT_STIM=.008 -D RAMPUPRATE -c src/c/rampUp.c -o rampUpRate.o
	gcc  rampUpRate.o $(LIBS) -o bin/rampUpRate
	rm rampUpRate.o
#	time ./bin/rampUpRate

# same as above, but now also with CURRENTPRED
rampUpRateC:
	gcc  $(FLAGS) -D FILENAME_POST=\"data/rampUpRatePostC.dat\" -D FILENAME_PRE=\"data/rampUpRatePreC.dat\" -D RAMPUPRATE -D CURRENTPRED -D INPUT_AT_STIM=.008 -c src/c/rampUp.c -o rampUpRateC.o
	gcc  rampUpRateC.o $(LIBS) -o bin/rampUpRateC
	rm rampUpRateC.o
#	time ./bin/rampUpRateC

# another simulation based on rampUp.c, 
rotations:

	@for i in $$(seq 20 10 200); do \
		gcc $(FLAGS) -D ROTATIONS -D RECORD_PREACT -D TAU_ALPHA=$$i -D FILENAME_POST=\"data/rotationsPost_$$i.dat\" -D FILENAME_PRE=\"data/rotationsPre_$$i.dat\" -D FROZEN_POISSON -c src/c/rampUp.c -o rotation$$i.o; gcc rotation$$i.o $(LIBS) -o bin/rotation$$i; rm rotation$$i.o; \
#		time ./bin/rotation$$i; \
	done

# xxx next experiments xxx -- think about the prospective coding

traceConditioning:
	gcc  -std=c99 -O3 -D FILENAME=\"data/conditioning.dat\" -D TAU_ALPHA=2000 -c src/c/conditioning.c
	gcc  conditioning.o -lgsl -lgslcblas -lm -o bin/conditioning
	rm conditioning.o
#	time ./bin/conditioning

recurrent:
	gcc  -std=c99 -O3 -D TAU_ALPHA=40 -c src/c/recurrent.c;
	gcc recurrent.o -lgsl -lgslcblas -lm -o bin/recurrent
	rm recurrent.o
#	time ./bin/recurrent

makedirs:
	mkdir -p data bin

all: makedirs rampUp rampUpC rampUpFP rampUpFPC rampUpRate rampUpRateC rotations traceConditioning recurrent
