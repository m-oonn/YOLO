$url = "http://127.0.0.1:8080/"
$output = "E:\projects-YOLO\screenshot-homepage.png"

# Start Edge in headless mode and take screenshot
$edgePath = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
& $edgePath --headless --disable-gpu --screenshot="$output" --window-size=1920,1080 --virtual-time-budget=5000 "$url"

Start-Sleep -Seconds 3
if (Test-Path $output) {
    Write-Host "Screenshot saved to $output"
} else {
    Write-Host "Screenshot failed"
}
