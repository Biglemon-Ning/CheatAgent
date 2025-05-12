#!/bin/sh

# # for i in "ML1M" "Taobao" ;do
# for i in "ML1M" "Taobao" ;do
#     for j in "sequential" ;do
#         for attack_mode in "HA-WOG" ;do
#         # for attack_mode in "HA-WOG" ;do
#             echo Start $i\_$j.sh \'$attack_mode\' $(date "+%Y-%m-%d %H:%M:%S") >> result.txt
#             sh $i\_$j.sh $attack_mode 2,3 >> result.txt
#             echo End $i\_$j.sh \'$attack_mode\' $(date "+%Y-%m-%d %H:%M:%S") >> result.txt
#             echo '\n\n\n' >> result.txt
#         done
#     done
# done

# =================== Attack ======================
for i in "LastFM" "ML1M" "Taobao" ;do
    for j in "sequential" ;do
        # for attack_mode in "PI-N" "PI-T" "TextFooler" "GA" "RL" "BAE" "LLMBA" "RP" "RPFP" "FPRP" "PA" "RT" "RTFP" ;do
        for attack_mode in "HA" ;do
            echo Start $i\_$j.sh \'$attack_mode\' $(date "+%Y-%m-%d %H:%M:%S")
            sh $i\_$j.sh $attack_mode 0,1 "--write"
            # sh $i\_$j.sh $attack_mode 1,3
            echo End $i\_$j.sh \'$attack_mode\' $(date "+%Y-%m-%d %H:%M:%S")
            echo '\n\n\n'
        done
    done
done


# # =================== Defense ======================
# for i in "Taobao" ;do
#     for j in "sequential" ;do
#         # for attack_mode in "PI-N" "PI-T" "TextFooler" "GA" "RL" "BAE" "LLMBA" "RP" "RPFP" "FPRP" "PA" "RT" "RTFP" ;do
#         # for attack_mode in "PA" "TA" ;do
#         for attack_mode in "TA" ;do
#             # for defense_mode in "PD" "RPD" "RTD" "APD-D" ;do
#             # for defense_mode in "PD" "RPD" "RTD" ;do
#             for defense_mode in "APD-D" ;do
#             echo Start $i\_$j.sh \'$attack_mode\' \'$defense_mode\' $(date "+%Y-%m-%d %H:%M:%S")
#             sh $i\_$j.sh $attack_mode 1,3 "--defense --defense_mode $defense_mode --seed $(date "+%S")"
#             echo End $i\_$j.sh \'$attack_mode\' \'$defense_mode\' $(date "+%Y-%m-%d %H:%M:%S")
#             echo '\n\n\n'
#             done
#         done
#     done
# done