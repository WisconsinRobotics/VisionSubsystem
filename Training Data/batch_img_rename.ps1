param(
    [Parameter(Mandatory=$true)][string]$folderpath,
    [Parameter(Mandatory=$true)][string]$filelabel,
    [Parameter(Mandatory=$true)][int]$startnum
)

cd $folderpath
#get-childitem -filter "*images*" | foreach-object {
#    rename-item $_.fullname "$filelabel`_$startnum.jpg"
#    $startnum++
#}
get-childitem | foreach-object {
    rename-item $_.fullname "$filelabel`_$startnum.jpg"
    $startnum++
}
