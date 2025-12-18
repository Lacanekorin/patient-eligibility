# Benchmark script for testing all medical NLP models
# Usage: .\benchmark_models.ps1

$ErrorActionPreference = "Stop"

# List of models to test
$models = @(
    "pubmedbert",
    "pubmedbert-neuml",
    "sapbert",
    "bioclinicalbert",
    "clinicalbert",
    "biomedbert",
    "mpnet",
    "minilm"
)

# Test file
$testFile = "data/uploads/Batch1withGroundTruth.xlsx"

# Results file
$resultsFile = "data/results/benchmark_results.txt"

# Create results header
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"=" * 60 | Out-File $resultsFile
"Model Benchmark Results - $timestamp" | Out-File $resultsFile -Append
"=" * 60 | Out-File $resultsFile -Append
"" | Out-File $resultsFile -Append

Write-Host "Starting benchmark of $($models.Count) models..."
Write-Host "Test file: $testFile"
Write-Host ""

foreach ($model in $models) {
    Write-Host "=" * 60
    Write-Host "Testing model: $model"
    Write-Host "=" * 60

    # Record start time
    $startTime = Get-Date

    # Stop existing container
    docker-compose -f docker-compose.pytorch.yml down 2>$null

    # Set model type and start container
    $env:MODEL_TYPE = $model
    Write-Host "Starting container with MODEL_TYPE=$model..."
    docker-compose -f docker-compose.pytorch.yml up -d --build

    # Wait for container to be ready
    Write-Host "Waiting for service to be ready..."
    $maxWait = 120  # seconds
    $waited = 0
    $ready = $false

    while (-not $ready -and $waited -lt $maxWait) {
        Start-Sleep -Seconds 2
        $waited += 2
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:5000" -Method GET -TimeoutSec 5 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                $ready = $true
                Write-Host "Service ready after $waited seconds"
            }
        } catch {
            Write-Host "." -NoNewline
        }
    }

    if (-not $ready) {
        Write-Host ""
        Write-Host "ERROR: Service not ready after $maxWait seconds, skipping $model"
        "[$model] ERROR: Service not ready" | Out-File $resultsFile -Append
        continue
    }

    Write-Host ""
    Write-Host "Sending test file..."

    # Send test file via curl
    try {
        $curlOutput = & curl -s -X POST -F "file=@$testFile" http://localhost:5000/upload 2>&1

        # Extract accuracy from response
        if ($curlOutput -match "Accuracy.*?(\d+\.?\d*)%") {
            $accuracy = $matches[1]
        } elseif ($curlOutput -match "(\d+\.?\d*)\s*%.*accuracy") {
            $accuracy = $matches[1]
        } else {
            # Try to get from docker logs
            $logs = docker logs eligibility-checker 2>&1 | Select-String -Pattern "accuracy|Accuracy" | Select-Object -Last 1
            if ($logs -match "(\d+\.?\d*)") {
                $accuracy = $matches[1]
            } else {
                $accuracy = "N/A"
            }
        }

        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalSeconds

        Write-Host "Model: $model | Accuracy: $accuracy% | Time: $([math]::Round($duration, 1))s"
        "[$model] Accuracy: $accuracy% | Time: $([math]::Round($duration, 1))s" | Out-File $resultsFile -Append

    } catch {
        Write-Host "ERROR processing $model : $_"
        "[$model] ERROR: $_" | Out-File $resultsFile -Append
    }

    # Get docker logs for details
    Write-Host "Getting logs..."
    docker logs eligibility-checker 2>&1 | Select-Object -Last 30 | Out-File "data/results/logs_$model.txt"

    Write-Host ""
}

# Cleanup
Write-Host "Stopping containers..."
docker-compose -f docker-compose.pytorch.yml down

Write-Host ""
Write-Host "=" * 60
Write-Host "Benchmark complete! Results saved to: $resultsFile"
Write-Host "=" * 60
Get-Content $resultsFile
