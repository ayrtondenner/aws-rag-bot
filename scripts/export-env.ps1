param(
    [Parameter(Mandatory = $false)]
    [string]$EnvName = "aws-rag-bot",

    # Prefer a relative path so the exported environment.yml is portable.
    [Parameter(Mandatory = $false)]
    [string]$Prefix = "./.conda",

    [Parameter(Mandatory = $false)]
    [string]$OutFile = "environment.yml"
)

$ErrorActionPreference = "Stop"

function Normalize-PosixPath([string]$path) {
    return ($path -replace "\\", "/")
}

$normalizedPrefix = Normalize-PosixPath $Prefix

# If the target prefix exists, export that env; otherwise fall back to the currently active env.
# (conda errors if you pass -p to a non-existent prefix)
if (Test-Path $Prefix) {
    $exportLines = & conda env export --from-history -p $Prefix
}
else {
    $exportLines = & conda env export --from-history
}

$updatedLines = $exportLines | ForEach-Object {
    $_ 
} | ForEach-Object {
    if ($_ -match "^name:") {
        "name: $EnvName"
    }
    elseif ($_ -match "^prefix:") {
        "prefix: $normalizedPrefix"
    }
    else {
        $_
    }
}

# If the export didn't contain a prefix line (rare), append one.
if (-not ($updatedLines | Select-String -SimpleMatch "prefix:" -Quiet)) {
    $updatedLines += "prefix: $normalizedPrefix"
}

$updatedLines | Set-Content -Path $OutFile -Encoding utf8

Write-Host "Wrote $OutFile with name='$EnvName' and prefix='$normalizedPrefix'"