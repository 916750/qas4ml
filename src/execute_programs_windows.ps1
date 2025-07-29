# Define settings for the program executions
$program = "main.py"   # Path to the Python script to be executed

########################################################################
## For $program = "main.py"
####
$batch_mode = "fixed"

# Define whether to create plots or not
$create_plots = $true
########################################################################

# Define whether to send email or not when script execution finished
$send_mail = $true

if ($program -eq "main.py"){
    if ($batch_mode -eq "window") {
        $settings = @(
            @{ct='iris_2_1-2'; cn=2; rn=1; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(1, 9); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=2; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(1, 9); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=3; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(1, 9); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=1; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(10, 19); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=2; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(10, 19); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=3; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(10, 19); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=1; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(20, 29); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=2; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(20, 29); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=3; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(20, 29); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=1; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(30, 39); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=2; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(30, 39); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=3; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(30, 39); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=1; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(40, 49); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=2; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(40, 49); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=3; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(40, 49); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=1; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(50, 59); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=2; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(50, 59); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=3; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(50, 59); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=1; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(60, 69); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=2; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(60, 69); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=3; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(60, 69); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=1; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(70, 79); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=2; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(70, 79); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=3; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(70, 79); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=1; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(80, 89); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=2; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(80, 89); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=3; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(80, 89); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=1; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(90, 99); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=2; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(90, 99); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=3; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(90, 99); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=1; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(100, 105); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=2; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(100, 105); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=2; rn=3; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(100, 105); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
        )
    } elseif ($batch_mode -eq "list") {    
        $settings = @(  
            @{ct='iris_2_1-2'; cn=4; rn=1; cd=4; rm=0; ts=100000; ep=5; st=1024; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='list'; obs_win=@(1, 9); obs_list=5; obs_fix=101; br='false'; brt=0.95; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=4; rn=2; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='list'; obs_win=@(1, 9); obs_list=4; obs_fix=101; br='false'; brt=0.95; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
            @{ct='iris_2_1-2'; cn=4; rn=3; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='list'; obs_win=@(1, 9); obs_list=4; obs_fix=101; br='false'; brt=0.95; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
        )
    } elseif ($batch_mode -eq "fixed") {
        $settings = @(  
            # @{ct='mnist_2'; cn=17; rn=1; cd=6; rm=0; ts=100000; ep=4; st=1024; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='fixed'; obs_win=@(1, 9); obs_list=0; obs_fix=20; br='false'; brt=0.95; olr=0.01; opt='adam'; oep=1000; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='ppo'}
            # @{ct='mnist_2'; cn=17; rn=2; cd=6; rm=0; ts=100000; ep=4; st=1024; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='fixed'; obs_win=@(1, 9); obs_list=0; obs_fix=20; br='false'; brt=0.95; olr=0.01; opt='adam'; oep=1000; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='ppo'}
            @{ct='mnist_2'; cn=16; rn=3; cd=6; rm=0; ts=100000; ep=4; st=1024; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='fixed'; obs_win=@(1, 9); obs_list=0; obs_fix=20; br='false'; brt=0.95; olr=0.01; opt='adam'; oep=1000; gpm='random'; gpv=1.0; gps=3; mm='all'; alg='ppo'}
        )
    }
} elseif ($program -eq "analyse.py") {
    $settings = @(
        @{ct='iris_2_1-2'; cn=2; rn=1; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(1, 9); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
        @{ct='iris_2_1-2'; cn=2; rn=2; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(1, 9); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
        @{ct='iris_2_1-2'; cn=2; rn=3; cd=4; rm=0; ts=200000; ep=1; st=512; bs=128; lr=0.0003; ga=0.99; cr=0.2; ec=0.03; vf=0.5; na=0; bm='window'; obs_win=@(1, 9); obs_list=2; obs_fix=101; br='false'; brt=0.8; olr=0.01; opt='adam'; oep=200; gpm='random'; gpv=1.0; gps=3; mm='minimum'; alg='random'}
    )
}

# Progress tracking
$total_runs = $settings.Count
$current_run = 0
$start_total = Get-Date

foreach ($setting in $settings) {
    $current_run++
    $start_time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[Log] ${start_time}: Starting program execution ($current_run/$total_runs):"
    Write-Host "                           Arguments: " `
                "--cn $($setting.cn) --rn $($setting.rn) --cd $($setting.cd) --rm $($setting.rm) --ts $($setting.ts)" `
                "--ep $($setting.ep) --st $($setting.st) --bs $($setting.bs) --ga $($setting.ga) --lr $($setting.lr)" `
                "--cr $($setting.cr) --ec $($setting.ec) --vf $($setting.vf) --na $($setting.na) --bm $($setting.bm)" `
                "--obs_win $($setting.obs_win -join ' ') --obs_list $($setting.obs_list) --obs_fix $($setting.obs_fix)" `
                "--br $($setting.br) --brt $($setting.brt) --ct $($setting.ct)" `
                "--olr $($setting.olr) --opt $($setting.opt) --oep $($setting.oep)" `
                "--gpm $($setting.gpm) --gpv $($setting.gpv) --gps $($setting.gps) --mm $($setting.mm) --alg $($setting.alg)"
    
    $start = Get-Date

    # Execute the Python script
    python $program `
        --ct $($setting.ct) `
        --cn $($setting.cn) `
        --rn $($setting.rn) `
        --cd $($setting.cd) `
        --rm $($setting.rm) `
        --ts $($setting.ts) `
        --ep $($setting.ep) `
        --st $($setting.st) `
        --bs $($setting.bs) `
        --lr $($setting.lr) `
        --ga $($setting.ga) `
        --cr $($setting.cr) `
        --ec $($setting.ec) `
        --vf $($setting.vf) `
        --na $($setting.na) `
        --bm $($setting.bm) `
        --obs_win $($setting.obs_win[0]) $($setting.obs_win[1]) `
        --obs_list $($setting.obs_list) `
        --obs_fix $($setting.obs_fix) `
        --br $($setting.br) `
        --brt $($setting.brt) `
        --olr $($setting.olr) `
        --opt $($setting.opt) `
        --oep $($setting.oep) `
        --gpm $($setting.gpm) `
        --gpv $($setting.gpv) `
        --gps $($setting.gps) `
        --mm $($setting.mm) `
        --alg $($setting.alg) ` 
    $exit_code = $LASTEXITCODE

    $end = Get-Date
    $elapsed = ($end - $start)
    $elapsed_formatted = "{0:hh\:mm\:ss}" -f $elapsed
    $end_time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    if ($exit_code -eq 0) {
        Write-Host "[Log] ${end_time}: Program completed successfully"
    } else {
        Write-Host "[Log] ${end_time}: Program failed with exit code ${exit_code}"
    }
    Write-Host "                           Elapsed time: ${elapsed}s"
    Write-Host "------------------------------------------------------------------------------------------------------------------"
}

$time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

if ($create_plots) {
    Write-Host "[Log] ${time}: All program executions completed. Creating plots..."
    $first_setting = $settings[0]   
    python plot.py `
        --ct $($setting.ct) `
        --cn $($setting.cn) `
        --rn $($setting.rn) `
        --cd $($setting.cd) `
        --rm $($setting.rm) `
        --ts $($setting.ts) `
        --ep $($setting.ep) `
        --st $($setting.st) `
        --bs $($setting.bs) `
        --lr $($setting.lr) `
        --ga $($setting.ga) `
        --cr $($setting.cr) `
        --ec $($setting.ec) `
        --vf $($setting.vf) `
        --na $($setting.na) `
        --bm $($setting.bm) `
        --obs_win $($setting.obs_win[0]) $($setting.obs_win[1]) `
        --obs_list $($setting.obs_list) `
        --obs_fix $($setting.obs_fix) `
        --br $($setting.br) `
        --brt $($setting.brt) `
        --olr $($setting.olr) `
        --opt $($setting.opt) `
        --oep $($setting.oep) `
        --gpm $($setting.gpm) `
        --gpv $($setting.gpv) `
        --gps $($setting.gps) `
        --mm $($setting.mm) `
        --alg $($setting.alg)

    $time = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[Log] ${time}: All plots created successfully."
} else {
    Write-Host "[Log] ${time}: All program executions completed (creating plots disabled)."
}


####
# Send email notification when script execution finished
####

# Test the connection
# Test-NetConnection -ComputerName smtp.gmail.com -Port 465

if ($send_mail) {
    $smtpServer = "" # Your SMTP server
    # Port TLS 587; Port SSL 465
    $smtpPort = 587
    $smtpUser = "" # Your email address   
    $passwordFile = "" # Path to txt file containing your password 
    $recipient = "" # Recipient email address
    $subject = "BA script execution finished"
    $body = "Script execution for setting --cn $($setting.cn) --cd $($setting.cd) finished."

    try {
        $securePassword = Get-Content $passwordFile | ConvertTo-SecureString
        $smtpPassword = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword))
    } catch {
        Write-Host "[Log] ${time}: Reading or decrypting password failed: $_"
        exit 1
    }

    $mail = New-Object System.Net.Mail.MailMessage
    $mail.From = $smtpUser
    $mail.To.Add($recipient)
    $mail.Subject = $subject
    $mail.Body = $body

    $smtp = New-Object System.Net.Mail.SmtpClient($smtpServer, $smtpPort)
    $smtp.EnableSsl = $true
    $smtp.Timeout = 60000 # in milliseconds
    $smtp.Credentials = New-Object System.Net.NetworkCredential($smtpUser, $smtpPassword)

    try {
        $smtp.Send($mail)
        Write-Host "[Log] ${time}: Email notification sent to $smtpTo."
    } catch {
        Write-Host "[Log] ${time}: Error occurred while sending email: $_"
    }
}

$end_total = Get-Date
$elapsed_total = ($end_total - $start_total)
$elapsed_total = "{0:hh\:mm\:ss}" -f $elapsed_total
Write-Host "[Log] ${time}: Finished script execution after ${elapsed_total}."
