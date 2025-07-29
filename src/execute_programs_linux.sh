#!/bin/bash

program="main.py"   # Python file to be executed   

########################################################################
## For program="main.py"
####
# Define Batch mode for parameter optimization 
batch_mode="fixed"

# Define whether to create plots or not
create_plots=false
########################################################################

if [ "$program" == "main.py" ]; then
    if [ "$batch_mode" == "window" ]; then
        settings=(
            "--ct mnist --cn 1 --rn 1 --cd 10 --rm 0 --ts 100000 --ep 5 --st 2048 --bs 128 --lr 0.0003 --ga 0.99 --cr 0.2 --ec 0.03 --vf 0.5 --na 0 --bm window --obs_win 1 9 --obs_list 2 --obs_fix 20 --br false --brt 0.95 --olr 0.01 --opt adam --oep 1000 --gpm random --gpv 1.0 --gps 3 --mm minimum --alg ppo"
            # Add more settings as needed
        )
    elif [ "$batch_mode" == "list" ]; then
        settings=(
            "--ct mnist --cn 1 --rn 1 --cd 10 --rm 0 --ts 100000 --ep 5 --st 2048 --bs 128 --lr 0.0003 --ga 0.99 --cr 0.2 --ec 0.03 --vf 0.5 --na 0 --bm list --obs_win 1 9 --obs_list 2 --obs_fix 20 --br false --brt 0.95 --olr 0.01 --opt adam --oep 1000 --gpm random --gpv 1.0 --gps 3 --mm minimum --alg ppo"
            # Add more settings as needed
        )
    elif [ "$batch_mode" == "fixed" ]; then
        settings=(
            "--ct mnist --cn 1 --rn 1 --cd 10 --rm 0 --ts 100000 --ep 5 --st 2048 --bs 128 --lr 0.0003 --ga 0.99 --cr 0.2 --ec 0.03 --vf 0.5 --na 0 --bm fixed --obs_win 1 9 --obs_list 2 --obs_fix 20 --br false --brt 0.95 --olr 0.01 --opt adam --oep 1000 --gpm random --gpv 1.0 --gps 3 --mm minimum --alg ppo"
            # "--ct mnist_2 --cn 1 --rn 2 --cd 5 --rm 0 --ts 100000 --ep 1 --st 128 --bs 128 --lr 0.0003 --ga 0.99 --cr 0.2 --ec 0.03 --vf 0.5 --na 0 --bm fixed --obs_win 1 9 --obs_list 2 --obs_fix 16 --br false --brt 0.95 --olr 0.01 --opt adam --oep 1000 --gpm random --gpv 1.0 --gps 3 --mm minimum --alg ppo"
            # "--ct mnist_2 --cn 1 --rn 3 --cd 5 --rm 0 --ts 100000 --ep 1 --st 128 --bs 128 --lr 0.0003 --ga 0.99 --cr 0.2 --ec 0.03 --vf 0.5 --na 0 --bm fixed --obs_win 1 9 --obs_list 2 --obs_fix 16 --br false --brt 0.95 --olr 0.01 --opt adam --oep 1000 --gpm random --gpv 1.0 --gps 3 --mm minimum --alg ppo"
            # Add more settings as needed
        )
    fi
elif [ "$program" == "analyse" ]; then
    settings=(
            "--ct mnist --cn 1 --rn 1 --cd 10 --rm 0 --ts 100000 --ep 5 --st 2048 --bs 128 --lr 0.0003 --ga 0.99 --cr 0.2 --ec 0.03 --vf 0.5 --na 0 --bm fixed --obs_win 1 9 --obs_list 2 --obs_fix 20 --br false --brt 0.95 --olr 0.01 --opt adam --oep 1000 --gpm random --gpv 1.0 --gps 3 --mm minimum --alg ppo"
            # Add more settings as needed
    )
elif [ "$program" == "playground.py" ]; then
    settings=(
        "--ct mnist --cn 1 --rn 1 --cd 10 --rm 0 --ts 100000 --ep 5 --st 2048 --bs 128 --lr 0.0003 --ga 0.99 --cr 0.2 --ec 0.03 --vf 0.5 --na 0 --bm fixed --obs_win 1 9 --obs_list 2 --obs_fix 20 --br false --brt 0.95 --olr 0.01 --opt adam --oep 1000 --gpm random --gpv 1.0 --gps 3 --mm minimum --alg ppo"
            # Add more settings as needed
    )
fi

# Progress tracking
total_runs=${#settings[@]}
current_run=0
start_total=$(date +%s)

for setting in "${settings[@]}"; do
    ((current_run++))
    start_time=$(date +"%d-%m-%Y %H:%M:%S")
    echo "[Log] ${start_time}: Starting program execution ($current_run/$total_runs):"
    echo "                           Arguments: $setting"

    start=$(date +%s)

    python $program $setting

    exit_code=$?

    end=$(date +%s)
    elapsed=$((end - start))
    elapsed_formatted=$(printf "%02d:%02d:%02d" $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)))
    end_time=$(date +"%d-%m-%Y %H:%M:%S")

    if [ $exit_code -eq 0 ]; then
        echo "[Log] ${end_time}: Program completed successfully"
    else
        echo "[Log] ${end_time}: Program failed with exit code ${exit_code}"
    fi
    echo "                           Elapsed time: ${elapsed_formatted}s"
    echo "------------------------------------------------------------------------------------------------------------------"
done

time=$(date +"%d-%m-%Y %H:%M:%S")

if [ "$create_plots" == true ]; then
    echo "[Log] ${time}: All program executions completed. Creating plots..."
    
    first_setting=${settings[0]}
    python plot.py $first_setting
    
    time=$(date +"%d-%m-%Y %H:%M:%S")
    echo "[Log] ${time}: All plots created successfully."
else
    echo "[Log] ${time}: All program executions completed (creating plots disabled)."
fi


end_total=$(date +%s)
elapsed_total=$((end_total - start_total))
elapsed_total_formatted=$(printf "%02d:%02d:%02d" $((elapsed_total/3600)) $((elapsed_total%3600/60)) $((elapsed_total%60)))
echo "[Log] ${time}: Finished script execution after ${elapsed_total_formatted}."