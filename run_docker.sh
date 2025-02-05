#!/bin/bash
# Cambia al directorio 'docker'
cd docker || { echo "No se pudo cambiar al directorio 'docker'"; exit 1; }

# Ejecuta el contenedor Docker con las variables de entorno definidas en .env
docker run --env-file .env 970206120973.dkr.ecr.us-east-2.amazonaws.com/ept-vision-labeler 