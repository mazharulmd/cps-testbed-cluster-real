#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "helics/application_api/ValueFederate.hpp"
#include "json.hpp"
#include <algorithm>
#include <cstdlib>
#include <complex>
#include <cmath>
#include <set>
#include <vector>
#include <map>
#include <fstream>

using namespace ns3;
NS_LOG_COMPONENT_DEFINE("CpsTestbed");

static Ptr<Socket> g_pmuSocket;
static std::map<uint32_t,double> g_sendTime;
static std::map<uint32_t,double> g_recvV;
static std::map<uint32_t,double> g_trueV;     // actual value before any tampering
static std::map<uint32_t,double> g_delayMs;
static std::map<uint32_t,bool>   g_attacked;  // was this bus tampered?
static uint32_t g_rxCount = 0;
static double g_vLimit = 1.08;
static std::string g_pendingCommand;

// attack config
static bool     g_attackOn = false;
static uint32_t g_attackBus = 8;       // which bus to target
static double   g_fakeV = 1.05;
static uint32_t g_dropBus = 0;

void SendVoltage(uint32_t bus, double trueV) {
  if (g_dropBus != 0 && bus == g_dropBus) {
    std::cout << "[LOSS] bus " << bus << " packet dropped (targeted)" << std::endl;
    return;
  }
  double txV = trueV;
  bool tampered = false;
  // ---- FALSE DATA INJECTION: rewrite the targeted bus's value ----
  if (g_attackOn && bus == g_attackBus) {
    txV = g_fakeV;
    tampered = true;
  }
  g_trueV[bus] = trueV;
  g_attacked[bus] = tampered;

  std::ostringstream msg;
  msg << "bus=" << bus << ";V=" << txV;
  std::string s = msg.str();
  Ptr<Packet> pkt = Create<Packet>((uint8_t*)s.c_str(), s.size());
  g_sendTime[bus] = Simulator::Now().GetSeconds();
  g_pmuSocket->Send(pkt);
  std::cout << "[PMU-node0] sent bus=" << bus << " V=" << txV << " into network" << std::endl;
  if (tampered)
    std::cout << "[ATTACK] bus " << bus << " true V=" << trueV
              << " -> INJECTED FAKE V=" << txV << " (hiding violation)" << std::endl;
}

void PdcReceive(Ptr<Socket> socket) {
  Ptr<Packet> pkt;
  while ((pkt = socket->Recv())) {
    uint32_t sz = pkt->GetSize();
    std::vector<uint8_t> buf(sz);
    pkt->CopyData(buf.data(), sz);
    std::string s((char*)buf.data(), sz);
    uint32_t bus = 0; float v = 0;
    sscanf(s.c_str(), "bus=%u;V=%f", &bus, &v);
    double now = Simulator::Now().GetSeconds();
    g_rxCount++;
    g_recvV[bus] = v;
    g_delayMs[bus] = (now - g_sendTime[bus]) * 1000.0;
    std::cout << "[PDC-node1] received bus=" << bus << " V=" << v << " (delay " << g_delayMs[bus] << " ms)" << std::endl;
    // PDC decides based on what it RECEIVED (possibly tampered)
    if (v > g_vLimit) {
      std::ostringstream c; c << "REDUCE_VOLTAGE@bus" << bus;
      g_pendingCommand = c.str();
      std::cout << "[PDC] bus " << bus << " V=" << v
                << " -> VIOLATION detected, command raised" << std::endl;
    }
  }
}

int RunLegacy (int argc, char *argv[])
{
  double latencyMs = 10.0;
  double lossRate = 0.0;
  bool attack = false;
  std::string scenario = "run";
  CommandLine cmd;
  cmd.AddValue("latency", "link latency in ms", latencyMs);
  cmd.AddValue("loss", "packet loss rate 0..1", lossRate);
  cmd.AddValue("attack", "enable false data injection", attack);
  cmd.AddValue("name", "scenario name", scenario);
  uint32_t dropBus = 0;
  cmd.AddValue("droppmu", "drop a specific bus packet", dropBus);
  cmd.Parse (argc, argv);
  g_attackOn = attack;
  g_dropBus = dropBus;

  helics::FederateInfo fi;
  fi.coreType = helics::CoreType::ZMQ;
  fi.coreInitString = "--federates=1";
  fi.setProperty(HELICS_PROPERTY_TIME_DELTA, 1.0);
  auto fed = std::make_shared<helics::ValueFederate>("ns3_network", fi);
  auto sub = fed->registerSubscription("gridpack/bus1_voltage");
  auto cmdPub = fed->registerGlobalPublication<std::string>("ns3/control_command");

  NodeContainer nodes; nodes.Create(2);
  PointToPointHelper p2p;
  p2p.SetDeviceAttribute("DataRate", StringValue("10Mbps"));
  p2p.SetChannelAttribute("Delay", TimeValue(MilliSeconds(latencyMs)));
  NetDeviceContainer devs = p2p.Install(nodes);

  if (lossRate > 0.0) {
    Ptr<RateErrorModel> em = CreateObject<RateErrorModel>();
    em->SetAttribute("ErrorRate", DoubleValue(lossRate));
    em->SetAttribute("ErrorUnit", StringValue("ERROR_UNIT_PACKET"));
    devs.Get(1)->SetAttribute("ReceiveErrorModel", PointerValue(em));
  }

  InternetStackHelper stack; stack.Install(nodes);
  Ipv4AddressHelper addr; addr.SetBase("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer ifaces = addr.Assign(devs);

  uint16_t port = 5000;
  Ptr<Socket> pdcSocket = Socket::CreateSocket(nodes.Get(1),
      TypeId::LookupByName("ns3::UdpSocketFactory"));
  pdcSocket->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
  pdcSocket->SetRecvCallback(MakeCallback(&PdcReceive));

  g_pmuSocket = Socket::CreateSocket(nodes.Get(0),
      TypeId::LookupByName("ns3::UdpSocketFactory"));
  g_pmuSocket->Bind();
  g_pmuSocket->Connect(InetSocketAddress(ifaces.GetAddress(1), port));

  fed->enterExecutingMode();
  std::cout << "[CPS] scenario=" << scenario << " latency=" << latencyMs
            << "ms loss=" << (lossRate*100) << "% attack="
            << (attack ? "ON (FDI on bus 8)" : "OFF") << std::endl;

  uint32_t busNum = 0;
  std::map<uint32_t,double> sentTrueV;
  helics::Time t = 0.0;
  while (t < 15.0) {
    t = fed->requestTime(15.0);
    if (sub.isUpdated()) {
      busNum++;
      double v = sub.getDouble();
      sentTrueV[busNum] = v;
      Simulator::Schedule(Seconds(0.0), &SendVoltage, busNum, v);
      Simulator::Stop(MilliSeconds(latencyMs * 2 + 50));
      Simulator::Run();
      if (!g_pendingCommand.empty()) {
        cmdPub.publish(g_pendingCommand);
        std::cout << "[CONTROL] command -> GridPACK: " << g_pendingCommand << std::endl;
        g_pendingCommand.clear();
      } else cmdPub.publish("");
    }
  }
  fed->finalize();

  // ---- results CSV: true vs received (delivered), with attack flag ----
  // legacy demo results, read by the Node-RED console (CPS_SHARED/legacy)
  const char *shared = std::getenv ("CPS_SHARED");
  std::string base = std::string (shared ? shared : "/srv/cps") + "/legacy/";
  std::ofstream out(base + "last_compare.csv");
  out << "bus,sent_v,recv_v,delay_ms,delivered,attacked\n";
  uint32_t lost = 0, violations = 0, missed = 0;
  double delaySum = 0, delayMax = 0;
  for (uint32_t b = 1; b <= busNum; ++b) {
    double trueV = sentTrueV[b];
    out << b << "," << trueV << ",";
    bool atk = g_attacked.count(b) ? g_attacked[b] : false;
    if (g_recvV.count(b)) {
      double d = g_delayMs[b];
      out << g_recvV[b] << "," << d << ",1," << (atk?1:0) << "\n";
      delaySum += d; if (d > delayMax) delayMax = d;
      if (g_recvV[b] > g_vLimit) violations++;
      // a MISSED violation: true value is a violation but received value hid it
      if (trueV > g_vLimit && g_recvV[b] <= g_vLimit) missed++;
    } else { out << ",," << "0," << (atk?1:0) << "\n"; lost++; }
  }
  out.close();

  double avgDelay = (g_rxCount>0) ? delaySum/g_rxCount : 0;
  double lossPct = (busNum>0) ? 100.0*lost/busNum : 0;

  std::ofstream js(base + "last_summary.json");
  js << "{\n"
     << "  \"scenario\": \"" << scenario << "\",\n"
     << "  \"latency_ms\": " << latencyMs << ",\n"
     << "  \"attack\": \"" << (attack ? "FALSE DATA INJECTION (bus 8)" : "none") << "\",\n"
     << "  \"total_buses\": " << busNum << ",\n"
     << "  \"delivered\": " << g_rxCount << ",\n"
     << "  \"lost\": " << lost << ",\n"
     << "  \"loss_actual_pct\": " << lossPct << ",\n"
     << "  \"violations_seen_by_pdc\": " << violations << ",\n"
     << "  \"missed_violations\": " << missed << ",\n"
     << "  \"avg_delay_ms\": " << avgDelay << ",\n"
     << "  \"max_delay_ms\": " << delayMax << ",\n"
     << "  \"helics_status\": \"connected (2 federates synced)\",\n"
     << "  \"gridpack\": \"IEEE 14-bus power flow\",\n"
     << "  \"ns3\": \"PMU->PDC UDP over point-to-point\"\n"
     << "}\n";
  js.close();

  std::cout << "[CPS] SUMMARY: delivered " << g_rxCount << "/" << busNum
            << ", violations seen " << violations
            << ", MISSED violations (hidden by attack) " << missed << std::endl;
  Simulator::Destroy();
  return 0;
}

// ============================================================================
// Multi-PMU mode (--config=<file>)
//
// PMUs at the optimally placed buses sample the phasors published by the grid
// federate at a fixed reporting rate and send them over their own point-to-point
// link (own latency, jitter and loss) to a single PDC at the control center. The
// PDC time-aligns frames and releases a measurement set when every PMU has
// reported or its wait window expires. Released sets go to the control-center
// federate; its commands travel back over the network to generator nodes and
// are then handed to the grid federate. HELICS time advances one frame at a time.
// ============================================================================
namespace mp
{
using json = nlohmann::json;
using cplx = std::complex<double>;

struct Pmu
{
  uint32_t id, bus, nch;
  double delayMs;
  Ptr<Socket> sock;
};

struct Rec
{
  double sent = -1, arr = -1;
  std::string status = "not_sent";
  bool attacked = false;
};

struct Buf
{
  std::map<size_t, std::pair<double, json>> data; // pmu index -> (arrival, phasors)
  bool released = false;
  bool scheduled = false;
};

static double rate = 30, pdcWaitS = 0.02, procMs = 2, jitterMs = 2, noiseMag = 0.001, noiseAng = 0.001;
static std::vector<Pmu> pmus;
static std::map<uint32_t, size_t> byId;
static std::vector<std::vector<cplx>> state;
static bool haveState = false;
static std::map<uint64_t, Buf> bufs;
static std::map<std::pair<uint64_t, uint32_t>, Rec> recs;
static json outSets = json::array(), outCmds = json::array(), outGenerators = json::array();
static json generatorState = json::array();
static std::map<uint32_t, Ptr<Socket>> genFeedbackSock;
static double telemetryS = 2.0;
static uint64_t telemetrySent = 0, telemetryDelivered = 0, acksSent = 0, acksDelivered = 0;
static Ptr<NormalRandomVariable> noise;
static Ptr<UniformRandomVariable> uni;
static Ptr<Socket> ccCmdSock;
static std::map<uint32_t, Ipv4Address> genAddr;
static uint64_t cmdSent = 0, cmdDelivered = 0;

// running counters for the live view (published on ns3/status twice per simulated second)
struct Stat
{
  uint64_t sent = 0, delivered = 0, late = 0, dropped = 0, attacked = 0;
  double lastLatMs = -1;
};
static std::vector<Stat> stats;                      // per PMU, same order as pmus
static uint64_t setsReleased = 0, setsComplete = 0;
static double releaseLatSum = 0;                     // s, over released sets

// attack: applied to frames of the target PMUs between start and end
static std::string atkType = "none";
static double atkStart = 0, atkEnd = 0, atkDelayMs = 0;
static std::set<uint32_t> atkBuses;
static std::map<uint32_t, std::vector<std::pair<uint32_t, cplx>>> atkDeltas;
// adaptive FDI: sensitivities h per channel; c is recomputed from the target's voltage
static std::map<uint32_t, std::vector<std::pair<uint32_t, cplx>>> atkCoef;
static int64_t atkRefBus = -1;
static uint32_t atkRefCh = 0;
static double atkVFake = 0;
static double atkRecordS = 0, atkRampS = 0, atkVset = 1.12;
static std::set<uint32_t> atkGenBuses;
// Noisy phasors only: each replay gets the current frame index/time at Sample().
static std::map<uint32_t, std::vector<std::vector<cplx>>> atkRecording;
static std::map<uint32_t, size_t> atkReplayPos;
static uint64_t replayed = 0, cmdsDropped = 0, cmdsDelayed = 0, cmdsForged = 0;

void
Release (uint64_t k)
{
  Buf &b = bufs[k];
  if (b.released)
    return;
  b.released = true;
  json pm = json::object ();
  for (auto &[i, v] : b.data)
    {
      pm[std::to_string (pmus[i].bus)] = {{"arr", v.first}, {"ph", v.second}};
      Rec &r = recs[{k, pmus[i].id}];
      r.arr = v.first;
      r.status = "delivered";
      stats[i].delivered++;
      stats[i].lastLatMs = (v.first - k / rate) * 1000;
    }
  setsReleased++;
  if (b.data.size () == pmus.size ())
    setsComplete++;
  releaseLatSum += Simulator::Now ().GetSeconds () - k / rate;
  outSets.push_back ({{"k", k}, {"t", k / rate}, {"release_t", Simulator::Now ().GetSeconds ()},
                      {"expected", pmus.size ()}, {"pmus", pm}});
}

void
PdcReceive (Ptr<Socket> socket)
{
  Ptr<Packet> pkt;
  while ((pkt = socket->Recv ()))
    {
      std::string s (pkt->GetSize (), '\0');
      pkt->CopyData ((uint8_t *) s.data (), s.size ());
      json f = json::parse (s, nullptr, false);
      if (f.is_discarded () || !byId.count (f["id"].get<uint32_t> ()))
        continue;
      size_t i = byId[f["id"].get<uint32_t> ()];
      uint64_t k = f["k"].get<uint64_t> ();
      Buf &b = bufs[k];
      if (b.released)
        {
          recs[{k, pmus[i].id}].status = "late";
          stats[i].late++;
          continue;
        }
      b.data[i] = {Simulator::Now ().GetSeconds (), f["ph"]};
      if (b.data.size () == pmus.size ())
        Release (k);
      else if (!b.scheduled)
        {
          b.scheduled = true;
          Simulator::Schedule (Seconds (pdcWaitS), &Release, k);
        }
    }
}

void
SendFrame (size_t i, uint64_t k, std::string payload)
{
  Ptr<Packet> pkt = Create<Packet> ((const uint8_t *) payload.data (), payload.size ());
  pmus[i].sock->Send (pkt);
  Rec &r = recs[{k, pmus[i].id}];
  r.sent = Simulator::Now ().GetSeconds ();
  r.status = "sent";
  stats[i].sent++;
}

// every PMU samples the latest grid state at the frame time k / rate
void
Sample (uint64_t k)
{
  double t = k / rate;
  bool atkOn = atkType != "none" && t >= atkStart && t < atkEnd;
  bool fdi = atkType == "fdi" || atkType == "coordinated";
  double scale = atkRampS > 0 ? std::min (1.0, std::max (0.0, (t - atkStart) / atkRampS)) : 1.0;
  cplx c (0, 0);
  bool adaptive = false;
  if (atkOn && fdi && atkRefBus >= 0)
    for (size_t i = 0; i < pmus.size (); ++i)
      if (pmus[i].bus == uint32_t (atkRefBus) && atkRefCh < state[i].size ())
        {
          cplx v = state[i][atkRefCh];
          c = std::polar (atkVFake - std::abs (v), std::arg (v));
          adaptive = true;
        }
  for (size_t i = 0; i < pmus.size (); ++i)
    {
      Pmu &p = pmus[i];
      std::vector<cplx> ph = state[i];
      for (cplx &c : ph)
        c = std::polar (std::abs (c) * (1 + noiseMag * noise->GetValue ()),
                        std::arg (c) + noiseAng * noise->GetValue ());
      if (atkType == "replay" && atkBuses.count (p.bus) && t >= atkStart - atkRecordS && t < atkStart)
        atkRecording[p.bus].push_back (ph);
      bool attacked = atkOn && atkBuses.count (p.bus);
      if (atkType == "replay" && atkRecording[p.bus].empty ())
        attacked = false; // a sub-frame recording window may contain no samples
      double extraMs = 0;
      if (attacked)
        {
          recs[{k, p.id}].attacked = true;
          stats[i].attacked++;
          if (atkType == "drop")
            {
              recs[{k, p.id}].status = "dropped_by_attack";
              stats[i].dropped++;
              continue;
            }
          if (fdi && adaptive)
            for (auto &[ch, h] : atkCoef[p.bus])
              if (ch < ph.size ())
                ph[ch] += scale * h * c;
          if (fdi && !adaptive)
            for (auto &[ch, d] : atkDeltas[p.bus])
              if (ch < ph.size ())
                ph[ch] += scale * d;
          if (atkType == "replay")
            {
              auto &recording = atkRecording[p.bus];
              ph = recording[atkReplayPos[p.bus]++ % recording.size ()];
              replayed++;
            }
          if (atkType == "delay")
            extraMs = atkDelayMs;
        }
      json arr = json::array ();
      for (cplx &c : ph)
        arr.push_back ({c.real (), c.imag ()});
      json f = {{"id", p.id}, {"k", k}, {"ph", arr}};
      double delayMs = procMs + jitterMs * uni->GetValue () + extraMs;
      Simulator::Schedule (MilliSeconds (delayMs), &SendFrame, i, k, f.dump ());
    }
}

void
GenReceive (uint32_t bus, Ptr<Socket> socket)
{
  Ptr<Packet> pkt;
  while ((pkt = socket->Recv ()))
    {
      std::string s (pkt->GetSize (), '\0');
      pkt->CopyData ((uint8_t *) s.data (), s.size ());
      json c = json::parse (s, nullptr, false);
      if (c.is_discarded ())
        continue;
      c["delivered_t"] = Simulator::Now ().GetSeconds ();
      outCmds.push_back (c);
      cmdDelivered++;
    }
}

// Reverse generator traffic uses the same point-to-point links as commands.
// Command attacks deliberately intercept only CC -> generator traffic.
void
CcGeneratorReceive (Ptr<Socket> socket)
{
  Ptr<Packet> pkt;
  while ((pkt = socket->Recv ()))
    {
      std::string s (pkt->GetSize (), '\0');
      pkt->CopyData ((uint8_t *) s.data (), s.size ());
      json packet = json::parse (s, nullptr, false);
      if (packet.is_discarded ())
        continue;
      packet["delivered_t"] = Simulator::Now ().GetSeconds ();
      if (packet.value ("type", std::string ()) == "telemetry")
        telemetryDelivered++;
      else if (packet.value ("type", std::string ()) == "ack")
        acksDelivered++;
      else
        continue;
      outGenerators.push_back (packet);
    }
}

void
SendGeneratorPacket (json packet)
{
  uint32_t bus = packet["gen_bus"].get<uint32_t> ();
  if (!genFeedbackSock.count (bus))
    return;
  std::string s = packet.dump ();
  genFeedbackSock[bus]->Send (Create<Packet> ((const uint8_t *) s.data (), s.size ()));
  if (packet.value ("type", std::string ()) == "telemetry")
    telemetrySent++;
  else
    acksSent++;
}

void
SampleGenerators ()
{
  // Use the latest grid-reported setpoint/Q, with its original sample time.
  for (auto packet : generatorState)
    {
      packet["type"] = "telemetry";
      SendGeneratorPacket (packet);
    }
  Simulator::Schedule (Seconds (telemetryS), &SampleGenerators);
}

void
TransmitCommand (json cmd)
{
  uint32_t bus = cmd["gen_bus"].get<uint32_t> ();
  if (!genAddr.count (bus))
    return;
  std::string s = cmd.dump ();
  ccCmdSock->SendTo (Create<Packet> ((const uint8_t *) s.data (), s.size ()), 0,
                     InetSocketAddress (genAddr[bus], 5000));
}

void
SendCommand (json cmd)
{
  uint32_t bus = cmd["gen_bus"].get<uint32_t> ();
  if (!genAddr.count (bus))
    return;
  cmdSent++;
  double t = Simulator::Now ().GetSeconds ();
  if (t >= atkStart && t < atkEnd && atkGenBuses.count (bus))
    {
      if (atkType == "cmd-drop")
        {
          cmdsDropped++;
          return;
        }
      if (atkType == "cmd-delay")
        {
          cmdsDelayed++;
          Simulator::Schedule (Seconds (atkDelayMs / 1000.0), &TransmitCommand, cmd);
          return;
        }
    }
  TransmitCommand (cmd);
}

void
ForgeCommands ()
{
  // Attacker impersonates the CC using its source socket. The forged packet
  // traverses the normal UDP generator link; cc/commands never contains it.
  for (uint32_t bus : atkGenBuses)
    {
      if (!genAddr.count (bus))
        continue;
      json cmd = {{"id", "attack:forged:" + std::to_string (bus)}, {"gen_bus", bus},
                  {"vset", atkVset}, {"issued_t", atkStart}, {"reason", "attack: forged command"}};
      cmdsForged++;
      SendCommand (cmd);
    }
}

double
Percentile (std::vector<double> v, double q)
{
  if (v.empty ())
    return 0;
  std::sort (v.begin (), v.end ());
  return v[std::min (v.size () - 1, size_t (q * (v.size () - 1) + 0.5))];
}

int
Run (const std::string &cfgPath)
{
  std::ifstream in (cfgPath);
  json cfg = json::parse (in);
  std::string outdir = cfg["outdir"];
  double duration = cfg.value ("duration", 10.0);
  rate = cfg.value ("rate", 30.0);
  telemetryS = cfg.value ("telemetry_s", 2.0);
  NS_ABORT_MSG_IF (!std::isfinite (telemetryS) || telemetryS <= 0, "telemetry_s must be finite and positive");
  pdcWaitS = cfg.value ("pdc_wait_ms", 20.0) / 1000.0;
  procMs = cfg.value ("proc_ms", 2.0);
  jitterMs = cfg.value ("jitter_ms", 2.0);
  double loss = cfg.value ("loss", 0.0);
  noiseMag = cfg["noise"].value ("mag", 0.001);
  noiseAng = cfg["noise"].value ("ang", 0.001);
  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (cfg.value ("seed", 1));

  json atk = cfg.value ("attack", json::object ());
  atkType = atk.value ("type", "none");
  atkStart = atk.value ("start", 0.0);
  atkEnd = atk.value ("end", 0.0);
  atkDelayMs = atk.value ("delay_ms", 0.0);
  atkRecordS = atk.value ("record_s", std::min (atkStart, atkEnd - atkStart));
  atkRampS = atk.value ("ramp_s", 0.0);
  atkVset = atk.value ("vset", 1.12);
  for (auto &b : atk.value ("gen_targets", json::array ()))
    atkGenBuses.insert (b.get<uint32_t> ());
  for (auto &b : atk.value ("targets", json::array ()))
    atkBuses.insert (b.get<uint32_t> ());
  json deltas = atk.value ("deltas", json::object ());
  for (auto &[bus, list] : deltas.items ())
    for (auto &d : list)
      {
        atkDeltas[std::stoul (bus)].push_back ({d["ch"].get<uint32_t> (), cplx (d["dre"], d["dim"])});
        atkCoef[std::stoul (bus)].push_back ({d["ch"].get<uint32_t> (), cplx (d.value ("hre", 0.0), d.value ("him", 0.0))});
      }
  if (atk.contains ("ref") && atk["ref"].is_object ())
    {
      atkRefBus = atk["ref"]["bus"].get<int64_t> ();
      atkRefCh = atk["ref"]["ch"].get<uint32_t> ();
      atkVFake = atk.value ("v_fake", 1.0);
    }

  // topology: star of PMU and generator nodes around the control-center node
  NodeContainer cc;
  cc.Create (1);
  NodeContainer pmuNodes, genNodes;
  pmuNodes.Create (cfg["pmus"].size ());
  genNodes.Create (cfg["gens"].size ());
  InternetStackHelper stack;
  stack.Install (cc);
  stack.Install (pmuNodes);
  stack.Install (genNodes);
  Ipv4AddressHelper addr;
  addr.SetBase ("10.0.0.0", "255.255.255.252");
  std::string accessRate = cfg.value ("access_rate", "10Mbps");
  int64_t stream = 100;

  noise = CreateObject<NormalRandomVariable> ();
  uni = CreateObject<UniformRandomVariable> ();
  noise->SetStream (1);
  uni->SetStream (2);

  Ptr<Socket> pdcSock = Socket::CreateSocket (cc.Get (0), UdpSocketFactory::GetTypeId ());
  pdcSock->Bind (InetSocketAddress (Ipv4Address::GetAny (), 4713));
  pdcSock->SetRecvCallback (MakeCallback (&PdcReceive));

  for (size_t i = 0; i < cfg["pmus"].size (); ++i)
    {
      json &pc = cfg["pmus"][i];
      Pmu p{pc["id"], pc["bus"], pc["nch"], pc["delay_ms"], nullptr};
      PointToPointHelper p2p;
      p2p.SetDeviceAttribute ("DataRate", StringValue (accessRate));
      p2p.SetChannelAttribute ("Delay", TimeValue (MicroSeconds (int64_t (p.delayMs * 1000))));
      NetDeviceContainer d = p2p.Install (pmuNodes.Get (i), cc.Get (0));
      if (loss > 0)
        {
          Ptr<RateErrorModel> em = CreateObject<RateErrorModel> ();
          em->SetAttribute ("ErrorRate", DoubleValue (loss));
          em->SetAttribute ("ErrorUnit", StringValue ("ERROR_UNIT_PACKET"));
          em->AssignStreams (stream++);
          d.Get (1)->SetAttribute ("ReceiveErrorModel", PointerValue (em));
        }
      Ipv4InterfaceContainer ifc = addr.Assign (d);
      addr.NewNetwork ();
      p.sock = Socket::CreateSocket (pmuNodes.Get (i), UdpSocketFactory::GetTypeId ());
      p.sock->Bind ();
      p.sock->Connect (InetSocketAddress (ifc.GetAddress (1), 4713));
      byId[p.id] = pmus.size ();
      pmus.push_back (p);
      state.push_back (std::vector<cplx> (p.nch, cplx (0, 0)));
    }

  Ptr<Socket> genRx = Socket::CreateSocket (cc.Get (0), UdpSocketFactory::GetTypeId ());
  genRx->Bind (InetSocketAddress (Ipv4Address::GetAny (), 5001));
  genRx->SetRecvCallback (MakeCallback (&CcGeneratorReceive));

  for (size_t g = 0; g < cfg["gens"].size (); ++g)
    {
      json &gc = cfg["gens"][g];
      uint32_t bus = gc["bus"];
      PointToPointHelper p2p;
      p2p.SetDeviceAttribute ("DataRate", StringValue (accessRate));
      p2p.SetChannelAttribute ("Delay", TimeValue (MicroSeconds (int64_t (gc["delay_ms"].get<double> () * 1000))));
      NetDeviceContainer d = p2p.Install (genNodes.Get (g), cc.Get (0));
      Ipv4InterfaceContainer ifc = addr.Assign (d);
      addr.NewNetwork ();
      genAddr[bus] = ifc.GetAddress (0);
      Ptr<Socket> feedback = Socket::CreateSocket (genNodes.Get (g), UdpSocketFactory::GetTypeId ());
      feedback->Bind ();
      feedback->Connect (InetSocketAddress (ifc.GetAddress (1), 5001));
      genFeedbackSock[bus] = feedback;
      Ptr<Socket> s = Socket::CreateSocket (genNodes.Get (g), UdpSocketFactory::GetTypeId ());
      s->Bind (InetSocketAddress (Ipv4Address::GetAny (), 5000));
      s->SetRecvCallback (MakeBoundCallback (&GenReceive, bus));
    }
  ccCmdSock = Socket::CreateSocket (cc.Get (0), UdpSocketFactory::GetTypeId ());
  ccCmdSock->Bind ();
  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();
  if (atkType == "cmd-inject" || atkType == "coordinated")
    Simulator::Schedule (Seconds (atkStart), &ForgeCommands);

  // HELICS federation: grid measurements in, aligned PDC sets out; commands both ways
  helics::FederateInfo fi;
  fi.coreType = helics::CoreType::ZMQ;
  // on a cluster the broker runs on another node: "--federates=1 --broker_address=tcp://head:23500 ..."
  fi.coreInitString = cfg.value ("helics_core_init", std::string ("--federates=1"));
  fi.setFlagOption (HELICS_FLAG_UNINTERRUPTIBLE, true);
  auto fed = std::make_shared<helics::ValueFederate> ("ns3_network", fi);
  auto subMeas = fed->registerSubscription ("grid/meas");
  auto subCmd = fed->registerSubscription ("cc/commands");
  auto subGenerators = fed->registerSubscription ("grid/generators");
  auto pubSets = fed->registerGlobalPublication<std::string> ("ns3/pdc");
  auto pubCmd = fed->registerGlobalPublication<std::string> ("ns3/cmd_delivered");
  auto pubGenerators = fed->registerGlobalPublication<std::string> ("ns3/generators");
  // what happens inside the network, for the dashboard's live view (no federate depends on it)
  auto pubStatus = fed->registerGlobalPublication<std::string> ("ns3/status");
  stats.assign (pmus.size (), Stat ());
  const uint64_t statusEvery = std::max<uint64_t> (1, uint64_t (std::llround (rate / 2)));
  fed->enterExecutingMode ();
  std::cout << "[NS3] " << pmus.size () << " PMUs -> PDC at " << rate << " frames/s, loss " << loss
            << ", attack " << atkType << std::endl;

  // HELICS can report an input as updated again without a new publication, so the
  // last payload handled is remembered and repeats are ignored
  std::string lastMeas, lastCmds, lastGenerators;
  bool telemetryStarted = false;
  uint64_t K = uint64_t (std::llround (duration * rate));
  for (uint64_t k = 0; k <= K; ++k)
    {
      fed->requestTime (k / rate);
      // publish what the previous interval produced, now that HELICS time has caught up
      if (!outSets.empty ())
        {
          pubSets.publish (outSets.dump ());
          outSets = json::array ();
        }
      if (!outCmds.empty ())
        {
          pubCmd.publish (outCmds.dump ());
          outCmds = json::array ();
        }
      if (!outGenerators.empty ())
        {
          pubGenerators.publish (outGenerators.dump ());
          outGenerators = json::array ();
        }
      if (subGenerators.isUpdated () && subGenerators.getString () != lastGenerators)
        {
          lastGenerators = subGenerators.getString ();
          json m = json::parse (lastGenerators, nullptr, false);
          if (!m.is_discarded ())
            {
              generatorState = m.value ("generators", json::array ());
              for (auto ack : m.value ("acks", json::array ()))
                {
                  ack["type"] = "ack";
                  SendGeneratorPacket (ack);
                }
              if (!telemetryStarted && !generatorState.empty ())
                {
                  telemetryStarted = true;
                  Simulator::ScheduleNow (&SampleGenerators);
                }
            }
        }
      if (subMeas.isUpdated () && subMeas.getString () != lastMeas)
        {
          lastMeas = subMeas.getString ();
          json m = json::parse (lastMeas, nullptr, false);
          if (!m.is_discarded ())
            {
              for (size_t i = 0; i < pmus.size (); ++i)
                {
                  auto it = m["pmus"].find (std::to_string (pmus[i].bus));
                  if (it == m["pmus"].end ())
                    continue;
                  for (size_t c = 0; c < it->size () && c < state[i].size (); ++c)
                    state[i][c] = cplx ((*it)[c][0], (*it)[c][1]);
                }
              haveState = true;
            }
        }
      if (subCmd.isUpdated () && subCmd.getString () != lastCmds)
        {
          lastCmds = subCmd.getString ();
          json cmds = json::parse (lastCmds, nullptr, false);
          if (cmds.is_array ())
            for (auto &c : cmds)
              SendCommand (c);
        }
      if (k % statusEvery == 0)
        {
          json pm = json::array ();
          for (size_t i = 0; i < pmus.size (); ++i)
            {
              const Stat &st = stats[i];
              pm.push_back ({pmus[i].bus, pmus[i].delayMs, st.sent, st.delivered, st.late, st.dropped,
                             st.attacked, std::round (st.lastLatMs * 100) / 100});
            }
          bool atkOn = atkType != "none" && k / rate >= atkStart && k / rate < atkEnd;
          pubStatus.publish (json ({{"t", k / rate},
                                    {"pmus", pm},
                                    {"pdc", {{"sets", setsReleased}, {"complete", setsComplete},
                                             {"release_ms", setsReleased ? 1000 * releaseLatSum / setsReleased : 0},
                                             {"waiting", bufs.size () - setsReleased}, {"wait_ms", pdcWaitS * 1000}}},
                                    {"commands", {{"sent", cmdSent}, {"delivered", cmdDelivered}}},
                                    {"attack", {{"type", atkType}, {"active", atkOn}}},
                                    {"events", Simulator::GetEventCount ()}})
                                 .dump ());
        }
      if (k == K)
        break;
      if (haveState)
        Sample (k);
      Simulator::Stop (Seconds ((k + 1) / rate) - Simulator::Now ());
      Simulator::Run ();
    }
  fed->finalize ();

  // ---- outputs: per-frame records and a summary ----
  std::ofstream fr (outdir + "/frames.csv");
  fr << "k,t,pmu_id,bus,sent_t,arr_t,latency_ms,status,attacked\n";
  std::map<std::string, uint64_t> counts;
  std::vector<double> lat;
  std::map<uint32_t, std::pair<uint64_t, uint64_t>> perPmu; // id -> (delivered, total)
  for (auto &[key, r] : recs)
    {
      if (r.status == "sent")
        r.status = "lost";
      if (r.status == "not_sent")
        continue;
      counts[r.status]++;
      double t = key.first / rate;
      double l = r.arr >= 0 ? (r.arr - t) * 1000 : -1;
      if (l >= 0)
        lat.push_back (l);
      auto &pp = perPmu[key.second];
      pp.second++;
      if (r.status == "delivered")
        pp.first++;
      fr << key.first << "," << t << "," << key.second << "," << pmus[byId[key.second]].bus << ","
         << r.sent << "," << r.arr << "," << l << "," << r.status << "," << (r.attacked ? 1 : 0) << "\n";
    }
  json per = json::object ();
  for (auto &[id, v] : perPmu)
    per[std::to_string (pmus[byId[id]].bus)] = double (v.first) / std::max<uint64_t> (v.second, 1);
  uint64_t total = 0;
  for (auto &[s, n] : counts)
    total += n;
  double mean = 0;
  for (double l : lat)
    mean += l;
  json summary = {{"frames_total", total},
                  {"status_counts", counts},
                  {"delivery_ratio", total ? double (counts["delivered"]) / total : 0.0},
                  {"latency_ms", {{"mean", lat.empty () ? 0 : mean / lat.size ()},
                                  {"p50", Percentile (lat, 0.5)},
                                  {"p95", Percentile (lat, 0.95)},
                                  {"max", Percentile (lat, 1.0)}}},
                  {"per_pmu_delivery", per},
                  {"commands_sent", cmdSent},
                  {"commands_delivered", cmdDelivered},
                  {"replayed", replayed},
                  {"cmds_dropped", cmdsDropped},
                  {"cmds_delayed", cmdsDelayed},
                  {"cmds_forged", cmdsForged},
                  {"telemetry_sent", telemetrySent},
                  {"telemetry_delivered", telemetryDelivered},
                  {"acks_sent", acksSent},
                  {"acks_delivered", acksDelivered}};
  std::ofstream(outdir + "/ns3_summary.json") << summary.dump (2);
  std::cout << "[NS3] done: " << counts["delivered"] << "/" << total << " frames delivered" << std::endl;
  Simulator::Destroy ();
  return 0;
}
} // namespace mp

int
main (int argc, char *argv[])
{
  for (int i = 1; i < argc; ++i)
    {
      std::string a = argv[i];
      if (a.rfind ("--config=", 0) == 0)
        return mp::Run (a.substr (9));
    }
  return RunLegacy (argc, argv);
}
