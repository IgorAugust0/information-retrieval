@echo off
REM usage: delete.bat
REM This batch file deletes some files

set found=0

REM List of files to delete
set files=resposta.txt response.txt indice.txt index.txt etiquetador.bin tagger.bin

REM Loop through the list of files and delete each file if it exists
for %%i in (%files%) do (
    if exist "%%i" (
        del "%%i"
        set found=1
    )
)

if %found%==0 (
    echo No files found to delete.
)

@REM ---- Alternative solutions ----
@REM -----------------------------------------------------------------------------
@REM REM usage: delete.bat
@REM REM This batch file deletes some files

@REM set found=0

@REM REM Check if the file "resposta.txt" or "response.txt" exists and delete it
@REM if exist resposta.txt (
@REM     del resposta.txt
@REM     set found=1
@REM ) else if exist response.txt (
@REM     del response.txt
@REM     set found=1
@REM )

@REM REM Check if the file "indice.txt" or "index.txt" exists and delete it
@REM if exist indice.txt (
@REM     del indice.txt
@REM     set found=1
@REM ) else if exist index.txt (
@REM     del index.txt
@REM     set found=1
@REM )

@REM REM Check if the file "etiquetador.bin" or "tagger.bin" exists and delete it
@REM if exist etiquetador.bin (
@REM     del etiquetador.bin
@REM     set found=1
@REM ) else if exist tagger.bin (
@REM     del tagger.bin
@REM     set found=1
@REM )

@REM if %found%==0 (
@REM     echo No files found to delete.
@REM )

@REM -----------------------------------------------------------------------------
@REM REM usage: delete.bat [file1] [file2] ...

@REM REM Redirect error output to a log file
@REM set LOGFILE=delete.log
@REM 2> %LOGFILE% (

@REM     REM Loop through command line arguments and delete each file
@REM     for %%i in (%*) do (
@REM         if exist "%%i" (
@REM             echo Deleting "%%i"...
@REM             del "%%i"
@REM         ) else (
@REM             echo File "%%i" not found.
@REM         )
@REM     )
@REM )

@REM REM Display log file
@REM type %LOGFILE%
@REM -----------------------------------------------------------------------------
