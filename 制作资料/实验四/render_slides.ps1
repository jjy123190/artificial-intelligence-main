param(
    [Parameter(Mandatory=$true)][string]$InputPath,
    [Parameter(Mandatory=$true)][string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
$presentationInput = (Resolve-Path -LiteralPath $InputPath).Path
[void](New-Item -ItemType Directory -Path $OutputDirectory -Force)
$presentationOutput = (Resolve-Path -LiteralPath $OutputDirectory).Path
$pptApplication = New-Object -ComObject PowerPoint.Application
$presentation = $null
try {
    $presentation = $pptApplication.Presentations.Open($presentationInput, $true, $false, $false)
    $textReview = @()
    for ($slideIndex = 1; $slideIndex -le $presentation.Slides.Count; $slideIndex++) {
        $slide = $presentation.Slides.Item($slideIndex)
        $slide.Export((Join-Path $presentationOutput ('slide-{0:D2}.png' -f $slideIndex)), 'PNG', 1920, 1080)
        foreach ($shape in $slide.Shapes) {
            if ($shape.HasTextFrame -and $shape.TextFrame.HasText) {
                $range = $shape.TextFrame.TextRange
                $textReview += [pscustomobject]@{
                    slide = $slideIndex
                    name = $shape.Name
                    text = $range.Text
                    left = $shape.Left
                    top = $shape.Top
                    width = $shape.Width
                    height = $shape.Height
                    text_width = $range.BoundWidth
                    text_height = $range.BoundHeight
                }
            }
        }
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($slide)
    }
    $textReview | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $presentationOutput 'text-fit.json') -Encoding UTF8
    Write-Output ('Rendered {0} PowerPoint slides.' -f $presentation.Slides.Count)
} finally {
    if ($null -ne $presentation) {
        $presentation.Close()
        [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($presentation)
    }
    $pptApplication.Quit()
    [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($pptApplication)
}
