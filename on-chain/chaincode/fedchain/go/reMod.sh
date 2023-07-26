rm go.mod
rm go.sum
rm -r vendor
go mod init
go mod vendor
chmod -R 777 ../go
