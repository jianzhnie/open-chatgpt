#!/bin/bash

echo  "kill all tasks with specified token"

if [ $# == 0 ]; then
    echo "Help: please specify the related token"
    echo "First parameter means the filtered string;"
    echo "Second parameter means excluding the specified string."
    exit 1;
elif [ $# == 1 ]; then
    ps aux | grep $1
elif [ $# == 2 ]; then
    ps aux | grep $1 | grep -v $2
else
  echo "please specify one or two related tokens"
fi

read -s -n1 -p "Are you sure killing them? (y/n)" pressed_key

if [ $pressed_key == "y" ]; then
    if [ $# == 1 ]; then
        ps aux | grep $1 | awk '{print $2}' | xargs kill -9
    elif [ $# == 2 ]; then
        ps aux | grep $1 | grep -v $2 | awk '{print $2}' | xargs kill -9
    fi
fi
echo
