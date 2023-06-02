#!/bin/bash


export LC_ALL=C  # ensure we are working with known locales
#set -e -u -f -o pipefail # safer shell script

declare -r PROGRAM=${0##*/}


if [ -z "$1" ]; then
  APPLICATION="ones-stream-full"
else
  APPLICATION="$1"
fi


declare -r DATADIR=./experiment_inputs/identification_inputs
declare -r OUTPUTDIR=./experiment_data/${APPLICATION}_identification
declare -r BENCHMARK='stream_c'
declare -r RUNNER='identification'
declare -r PARAMS_FILE='parameters.yaml'
declare -r ITERATION_COUNT='10_000'
declare -r PROBLEM_SIZE='33_554_432'
declare -r TOPOLOGY_FILE='topology.xml'

if [ ! -d "$OUTPUTDIR" ]; then
	mkdir -p "$OUTPUTDIR"
fi




declare -ra PRERUN_SNAPSHOT_FILES=(
        "${PARAMS_FILE}"
        "${TOPOLOGY_FILE}"
)

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

declare -a POSTRUN_SNAPSHOT_FILES=(
        # outputs
        dump_pubMeasurements.csv
        dump_pubProgress.csv
	identification-runner.log
        #nrm.log
        #time-metrics.csv
)


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
		benchmark: ${BENCHMARK}
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
	#if grep -q "step" ${cfg}; then
	if [[ ${cfg} == *"step"* ]]; then
	#	echo "sasi"
	#else
	#	echo "pakad"
	#fi
		timestamp="$(date --iso-8601=seconds)"
        	#archive="${OUTPUTDIR}/preliminaries_${BENCHMARK}_${timestamp}.tar"
		archive="${OUTPUTDIR}/preliminaries_${BENCHMARK}_${timestamp}.tar"
        	echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>${cfg}"
		dump_parameters "${timestamp}" "${RUNNER}" "${cfg}" "${BENCHMARK}" "--iterationCount=${ITERATION_COUNT} --problemSize=${PROBLEM_SIZE}"
		lstopo --output-format xml --whole-system --force "${OUTPUTDIR}/${TOPOLOGY_FILE}"
		tar --create --file="${archive}" --files-from=/dev/null
		tar --append --file="${archive}" --transform='s,^.*/,,' -- "${cfg}"
		tar --append --file="${archive}" --directory="${OUTPUTDIR}" -- "${PRERUN_SNAPSHOT_FILES[@]}"
		snapshot_system_state "${archive}" 'pre'
		python identification.py --enable-libnrm ${cfg} -- $APPLICATION 33554422 10000
		# retrieve benchmark logs and snapshot post-run state
		tar --append --file="${archive}" --directory="${OUTPUTDIR}" -- "${POSTRUN_SNAPSHOT_FILES[@]}"
		snapshot_system_state "${archive}" 'post'
		# compress archive
		xz --compress "${archive}"
		sleep 5
		python enforce_max_power.py max-range-config.yaml
        	sleep 10
		echo __________________________________________________________________________________________________
	fi

done
