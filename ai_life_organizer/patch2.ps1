$lines = Get-Content "app.py" -Encoding UTF8
$total = $lines.Count
Write-Host "Total lines: $total"

$startLine = 0
$endLine = 0
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i].StartsWith("BASE_TEMPLATE = ")) { $startLine = $i + 1; break }
}
for ($i = 0; $i -lt $lines.Count; $i++) {
    if ($lines[$i].StartsWith("REGISTER_TEMPLATE = ")) { $endLine = $i; break }
}
Write-Host "BASE_TEMPLATE: start=$startLine end=$endLine"

$newBase = Get-Content "_new_base.txt" -Encoding UTF8

$before = $lines[0..($startLine-2)]
$after  = $lines[($endLine-1)..($lines.Count-1)]

$newContent = ($before + $newBase + $after) -join "`n"
[System.IO.File]::WriteAllText("app.py", $newContent, [System.Text.Encoding]::UTF8)
Write-Host "Done. New line count: $(($newContent -split '\n').Count)"
