#!/bin/bash


export LC_ALL=C  # ensure we are working with known locales
#set -e -u -f -o pipefail # safer shell script

declare -r PROGRAM=${0##*/}


# parameters  -----------------------------------------------------------------

# xpctl subcommand for each supported runner
declare -rA RUNNERS=(
	[controller]="controller --controller-configuration"
	[identification]="identification --experiment-plan"
)

declare -r BENCHMARK='stream_c'
declare -r ITERATION_COUNT=10000


if [ -z "$1" ]; then
  APPLICATION="ones-stream-full"
else
  APPLICATION="$1"
fi

if [ "$APPLICATION" == "ones-npb-ep" ]; then
  declare -r PROBLEM_SIZE=22
elif [ "$APPLICATION" == "ones-solvers-cg" ]; then
  declare -r PROBLEM_SIZE=10000
elif [ "$APPLICATION" == "ones-solvers-bicgstab" ]; then
  declare -r PROBLEM_SIZE=10000
else
  declare -r PROBLEM_SIZE=33554432
fi


# configuration  --------------------------------------------------------------

declare -r LOGDIR='./experiment_data'  # all relative paths are relative to $LOGDIR
declare -r DATADIR="./experiment_inputs/control_SP/$APPLICATION"
declare -r OUTPUTDIR="./experiment_data/PI_control/$APPLICATION"


if [ ! -d "$OUTPUTDIR" ]; then
        mkdir -p "$OUTPUTDIR"
fi

declare -r PARAMS_FILE='parameters.yaml'
declare -r TOPOLOGY_FILE='topology.xml'


# files to snapshot before running the experiment
declare -ra PRERUN_SNAPSHOT_FILES=(
	"${PARAMS_FILE}"
	"${TOPOLOGY_FILE}"
)

# pseudo-files from /proc to record
declare -ra SYSTEM_STATE_SNAPSHOT_FILES=(
	/proc/cpuinfo
	/proc/iomem
	/proc/loadavg
	/proc/meminfo
	/proc/modules
	/proc/stat
	/proc/sys/kernel/hostname
	/proc/uptime
	/proc/version
	/proc/vmstat
	/proc/zoneinfo
)

# common (i.e., shared by runners) files to snapshot once the experiment is done
declare -a POSTRUN_SNAPSHOT_FILES=(
	# outputs
	dump_pubMeasurements.csv
	dump_pubProgress.csv
	controller-runner.log
	#nrm.log
	#time-metrics.csv
)

# runner-specific files to snapshot once the experiment is done
declare -rA RUNNERS_POSTRUN_SNAPSHOT_FILES=(
	[controller]="controller-runner.log"
	[identification]="identification-runner.log"
)

if [ ! -d "$OUTPUTDIR" ]; then
	mkdir -p "$OUTPUTDIR"
fi

# helper functions  -----------------------------------------------------------

function dump_parameters {
	declare -r timestamp="${1}"
	declare -r runner="${2}"
	declare -r cfg="${3}"
	declare -r benchmark="${4}"
	declare -r extra="${*:5}"

	cat <<- EOF > "${OUTPUTDIR}/${PARAMS_FILE}"
		timestamp: ${timestamp}
		runner: ${runner}
		config-file: ${cfg##*/}
		benchmark: ${benchmark}
		extra: ${extra}
	EOF
}


function snapshot_system_state {
	archive="${1}"
	subdir="${2}"

	# create unique namespace to work with
	wd=$(mktemp --directory)
	mkdir "${wd}/${subdir}"

	# snapshot
	for pseudofile in "${SYSTEM_STATE_SNAPSHOT_FILES[@]}"; do
		saveas="$(basename "${pseudofile}")"
		cat "${pseudofile}" > "${wd}/${subdir}/${saveas}"
	done

	# archive
	tar --append --file="${archive}" --directory="${wd}" --transform='s,^,sysstate/,' -- "${subdir}"

	# clean unique namespace
	rm --recursive --force -- "${wd}"
}


for cfg in "$DATADIR"/*
do
	if [[ ${cfg} == *"setpoint"* ]]; then
			timestamp="$(date --iso-8601=seconds)"
			archive="${OUTPUTDIR}/preliminaries_${BENCHMARK}_${timestamp}.tar"
			echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>${cfg}"
			dump_parameters "${timestamp}" "${RUNNER}" "${cfg}" "${BENCHMARK}" "--iterationCount=${ITERATION_COUNT} --problemSize=${PROBLEM_SIZE}"
			lstopo --output-format xml --whole-system --force "${OUTPUTDIR}/${TOPOLOGY_FILE}"
			tar --create --file="${archive}" --files-from=/dev/null
			tar --append --file="${archive}" --transform='s,^.*/,,' -- "${cfg}"
			tar --append --file="${archive}" --directory="${OUTPUTDIR}" -- "${PRERUN_SNAPSHOT_FILES[@]}"
			snapshot_system_state "${archive}" 'pre'
			if [ "$APPLICATION" == "ones-solvers-cg" ]; then
				python controller.py ${cfg} ones-solvers-cg 10000 poor 0 10000
			elif [ "$APPLICATION" == "ones-solvers-bicgstab" ]; then
				python controller.py ${cfg} ones-solvers-bicgstab 20000 poor 0 10000
			else
				python controller.py ${cfg} -- $APPLICATION $PROBLEM_SIZE $ITERATION_COUNT
			fi
			# retrieve benchmark logs and snapshot post-run state
			tar --append --file="${archive}" --directory="${OUTPUTDIR}" -- "${POSTRUN_SNAPSHOT_FILES[@]}"
			touch "${OUTPUTDIR}/SUCCESS"
			tar --append --file="${archive}" --directory="${OUTPUTDIR}" -- SUCCESS
			snapshot_system_state "${archive}" 'post'
			# compress archive
			xz --compress "${archive}"
	sleep 20
	python enforce_max_power.py max-range-config.yaml
	echo __________________________________________________________________________________________________
	sleep 20
	fi
done