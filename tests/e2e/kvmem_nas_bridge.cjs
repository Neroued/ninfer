// Run on the Windows GPU host while the NAS's EvalScope uses the WSL dev server.
// Only the explicitly selected NAS address can connect. No machine-wide forwarding
// or firewall rules are changed. Stop this process when the benchmark finishes.
'use strict';
const net = require('node:net');
const [listenAddress, listenValue, targetValue, allowedPeer] = process.argv.slice(2);
const listenPort = Number(listenValue), targetPort = Number(targetValue);
if (!net.isIPv4(listenAddress || '') || !net.isIPv4(allowedPeer || '') ||
    ![listenPort, targetPort].every(p => Number.isInteger(p) && p > 1024 && p < 65536 &&
                                       ![8080, 8081].includes(p))) {
  console.error('Usage: node kvmem_nas_bridge.cjs WINDOWS_LAN_IP LISTEN_PORT WSL_LOCAL_PORT NAS_IP');
  process.exit(2);
}
const sockets = new Set();
const server = net.createServer(client => {
  if (client.remoteAddress !== allowedPeer) { client.destroy(); return; }
  const upstream = net.connect({host: '127.0.0.1', port: targetPort});
  sockets.add(client); sockets.add(upstream);
  const close = () => {
    client.destroy(); upstream.destroy(); sockets.delete(client); sockets.delete(upstream);
  };
  client.on('error', close); upstream.on('error', close);
  client.on('close', close); upstream.on('close', close);
  client.pipe(upstream); upstream.pipe(client);
});
server.on('error', error => { console.error(error.code); process.exitCode = 1; });
server.listen(listenPort, listenAddress, () => console.log(JSON.stringify({
  pid: process.pid, listen: `${listenAddress}:${listenPort}`,
  target: `127.0.0.1:${targetPort}`, allowedPeer
})));
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => { for (const socket of sockets) socket.destroy(); server.close(); });
}
