<?php
    //Hier werden die Daten zur Verbindung gespeichert
    $server = "localhost";
    $database = "main";
    $user = "";
    $pwd = "G812brapla";
    //Hier wird die Verbindung aufgebaut,...
    try{
        //...das wird ausgeführt wenn es klappt
        $con = new PDO("mysql:host=$server;dbname=$database");
        $con->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        echo "<p>Successs</p>";
    }
    catch(PDOException $e){
        //...das wird ausgeführt wenn es einen Fehler gibt
        echo "<p>Es konnte keine Verbindung zur Datenbank hergestellt werden: " . $e->getMessage() . "</p>";
    }
?>