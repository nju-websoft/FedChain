#!/bin/bash
#
# Copyright IBM Corp All Rights Reserved
#
# SPDX-License-Identifier: Apache-2.0
#

ORG=$1
PORT=$2

function createOrg3 {
	infoln "Enrolling the CA admin"
	mkdir -p ../organizations/peerOrganizations/org${ORG}.example.com/

	export FABRIC_CA_CLIENT_HOME=${PWD}/../organizations/peerOrganizations/org${ORG}.example.com/

  set -x
  fabric-ca-client enroll -u https://admin:adminpw@localhost:${PORT} --caname ca-org${ORG} --tls.certfiles ${PWD}/fabric-ca/org${ORG}/tls-cert.pem
  { set +x; } 2>/dev/null

  echo "NodeOUs:
  Enable: true
  ClientOUIdentifier:
    Certificate: cacerts/localhost-${PORT}-ca-org${ORG}.pem
    OrganizationalUnitIdentifier: client
  PeerOUIdentifier:
    Certificate: cacerts/localhost-${PORT}-ca-org${ORG}.pem
    OrganizationalUnitIdentifier: peer
  AdminOUIdentifier:
    Certificate: cacerts/localhost-${PORT}-ca-org${ORG}.pem
    OrganizationalUnitIdentifier: admin
  OrdererOUIdentifier:
    Certificate: cacerts/localhost-${PORT}-ca-org${ORG}.pem
    OrganizationalUnitIdentifier: orderer" > ${PWD}/../organizations/peerOrganizations/org${ORG}.example.com/msp/config.yaml

	infoln "Registering peer0"
  set -x
	fabric-ca-client register --caname ca-org${ORG} --id.name peer0 --id.secret peer0pw --id.type peer --tls.certfiles ${PWD}/fabric-ca/org${ORG}/tls-cert.pem
  { set +x; } 2>/dev/null

  infoln "Registering user"
  set -x
  fabric-ca-client register --caname ca-org${ORG} --id.name user1 --id.secret user1pw --id.type client --tls.certfiles ${PWD}/fabric-ca/org${ORG}/tls-cert.pem
  { set +x; } 2>/dev/null

  infoln "Registering the org admin"
  set -x
  fabric-ca-client register --caname ca-org${ORG} --id.name org${ORG}admin --id.secret org${ORG}adminpw --id.type admin --tls.certfiles ${PWD}/fabric-ca/org${ORG}/tls-cert.pem
  { set +x; } 2>/dev/null

  infoln "Generating the peer0 msp"
  set -x
	fabric-ca-client enroll -u https://peer0:peer0pw@localhost:${PORT} --caname ca-org${ORG} -M ${PWD}/../organizations/peerOrganizations/org${ORG}.example.com/peers/peer0.org${ORG}.example.com/msp --csr.hosts peer0.org${ORG}.example.com --tls.certfiles ${PWD}/fabric-ca/org${ORG}/tls-cert.pem
  { set +x; } 2>/dev/null

  cp ${PWD}/../organizations/peerOrganizations/org${ORG}.example.com/msp/config.yaml ${PWD}/../organizations/peerOrganizations/org${ORG}.example.com/peers/peer0.org${ORG}.example.com/msp/config.yaml

  infoln "Generating the peer0-tls certificates"
  set -x
  fabric-ca-client enroll -u https://peer0:peer0pw@localhost:${PORT} --caname ca-org${ORG} -M ${PWD}/../organizations/peerOrganizations/org${ORG}.example.com/peers/peer0.org${ORG}.example.com/tls --enrollment.profile tls --csr.hosts peer0.org${ORG}.example.com --csr.hosts localhost --tls.certfiles ${PWD}/fabric-ca/org${ORG}/tls-cert.pem
  { set +x; } 2>/dev/null


  cp ${PWD}/../organizations/peerOrganizations/org${ORG}.example.com/peers/peer0.org${ORG}.example.com/tls/tlscacerts/* ${PWD}/../organizations/peerOrganizations/org${ORG}.example.com/peers/peer0.org${ORG}.example.com/tls/ca.crt
  cp ${PWD}/../organizations/peerOrganizations/org${ORG}.example.com/peers/peer0.org${ORG}.example.com/tls/signcerts/* ${PWD}/../organizations/peerOrganizations/org${ORG}.example.com/peers/peer0.org${ORG}.example.com/tls/server.crt
  cp ${PWD}/../organizations/peerOrganizations/org${ORG}.example.com/peers/peer0.org${ORG}.example.com/tls/keystore/* ${PWD}/../organizations/peerOrganizations/org${ORG}.example.com/peers/peer0.org${ORG}.example.com/tls/server.key

  mkdir ${PWD}/../organizations/peerOrganizations/org${ORG}.example.com/msp/tlscacerts
  cp ${PWD}/../organizations/peerOrganizations/org${ORG}.example.com/peers/peer0.org${ORG}.example.com/tls/tlscacerts/* ${PWD}/../organizations/peerOrganizations/org${ORG}.example.com/msp/tlscacerts/ca.crt

  mkdir ${PWD}/../organizations/peerOrganizations/org${ORG}.example.com/tlsca
  cp ${PWD}/../organizations/peerOrganizations/org${ORG}.example.com/peers/peer0.org${ORG}.example.com/tls/tlscacerts/* ${PWD}/../organizations/peerOrganizations/org${ORG}.example.com/tlsca/tlsca.org${ORG}.example.com-cert.pem

  mkdir ${PWD}/../organizations/peerOrganizations/org${ORG}.example.com/ca
  cp ${PWD}/../organizations/peerOrganizations/org${ORG}.example.com/peers/peer0.org${ORG}.example.com/msp/cacerts/* ${PWD}/../organizations/peerOrganizations/org${ORG}.example.com/ca/ca.org${ORG}.example.com-cert.pem

  infoln "Generating the user msp"
  set -x
	fabric-ca-client enroll -u https://user1:user1pw@localhost:${PORT} --caname ca-org${ORG} -M ${PWD}/../organizations/peerOrganizations/org${ORG}.example.com/users/User1@org${ORG}.example.com/msp --tls.certfiles ${PWD}/fabric-ca/org${ORG}/tls-cert.pem
  { set +x; } 2>/dev/null

  cp ${PWD}/../organizations/peerOrganizations/org${ORG}.example.com/msp/config.yaml ${PWD}/../organizations/peerOrganizations/org${ORG}.example.com/users/User1@org${ORG}.example.com/msp/config.yaml

  infoln "Generating the org admin msp"
  set -x
	fabric-ca-client enroll -u https://org${ORG}admin:org${ORG}adminpw@localhost:${PORT} --caname ca-org${ORG} -M ${PWD}/../organizations/peerOrganizations/org${ORG}.example.com/users/Admin@org${ORG}.example.com/msp --tls.certfiles ${PWD}/fabric-ca/org${ORG}/tls-cert.pem
  { set +x; } 2>/dev/null

  cp ${PWD}/../organizations/peerOrganizations/org${ORG}.example.com/msp/config.yaml ${PWD}/../organizations/peerOrganizations/org${ORG}.example.com/users/Admin@org${ORG}.example.com/msp/config.yaml
}
