#!/usr/bin/env bash
# check for the exps to run
for i in {1..2}; do
    echo python run.py -n "duddqn-$i" -dd -du -ra "$i"
    echo python run.py -n "dddqn-$i" -dd -ra "$i"
    echo python run.py -n "dudqn-$i" -du -ra "$i"
    echo python run.py -n "dqn-$i" -ra "$i"
done

# real run
for i in {1..5}; do
    python run.py -n "duddqn-$i" -dd -du -ra "$i"
    python run.py -n "dddqn-$i" -dd -ra "$i"
    python run.py -n "dudqn-$i" -du -ra "$i"
    python run.py -n "dqn-$i" -ra "$i"
done
