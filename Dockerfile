FROM maven:3.9.9-eclipse-temurin-17
WORKDIR /app

COPY pom.xml .
COPY src ./src

ARG MAVEN_PROFILE=
RUN mvn --errors --show-version --batch-mode --no-transfer-progress clean package ${MAVEN_PROFILE}
