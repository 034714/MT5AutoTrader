const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const { test } = require('node:test');

const html = fs.readFileSync(path.join(__dirname, '..', 'web', 'index.html'), 'utf8');
const start = html.indexOf('function srTradeBarIndex(');
const end = html.indexOf('function setSRTradeVisibility(', start);
assert(start >= 0 && end > start, 'Chart marker helpers must remain testable');
const context = vm.createContext({
  srChartFmt: timestamp => `time-${timestamp}`,
  srChartDec: () => 2,
});
vm.runInContext(html.slice(start, end), context);
const { srTradeBarIndex, srTradeMarkers, srNearestTradeMarker } = context;
const candles = [1000, 1060, 1120, 10000, 10060].map(time => [time, 100, 110, 90, 101]);
const layout = { start: 0, end: 5, padL: 8, padT: 12, plotW: 500, plotH: 300, barsVis: 5, lo: 80, hi: 120 };
const trade = {
  position_id: 42, direction: 'SELL', volume: 0.1,
  open_time: 10015, open_price: 102,
  closed: true, close_time: 10075, close_price: 100, close_volume: 0.1, profit: 2,
};
const build = (changes = {}, view = layout) => srTradeMarkers(candles, [{ ...trade, ...changes }], view, 60, 'EUR');

test('trade after a session gap maps to the actual candle', () => {
  assert.equal(srTradeBarIndex(candles, 10015, 60), 3);
  assert.equal(build().markers[0].x, 358);
});

test('timestamps in missing intervals or outside the loaded range are omitted', () => {
  for (const timestamp of [999, 1180, 9999, 10120, NaN]) {
    assert.equal(srTradeBarIndex(candles, timestamp, 60), null);
  }
  assert.equal(srTradeBarIndex([], 1000, 60), null);
});

test('exact candle boundaries select the newer candle', () => {
  assert.equal(srTradeBarIndex(candles, 1060, 60), 1);
  assert.equal(srTradeBarIndex(candles, 1059, 60), 0);
  assert.equal(srTradeBarIndex(candles, 10060, 60), 4);
});

test('entry before the loaded data does not hide a visible exit', () => {
  const result = build({ open_time: 500 });
  assert.equal(result.markers.length, 1);
  assert.equal(result.markers[0].shape, 'exit');
});

test('visible exits survive panning past the entry candle', () => {
  const result = build({}, { ...layout, start: 4, barsVis: 1 });
  assert.equal(result.markers.length, 1);
  assert.equal(result.markers[0].shape, 'exit');
});

test('price-out-of-range markers do not expand the chart or remain clickable', () => {
  const before = JSON.stringify(layout);
  assert.equal(build({ open_price: 160, close_price: 170 }).markers.length, 0);
  assert.equal(JSON.stringify(layout), before);
});

test('hover hit testing uses the actual triangle position', () => {
  const marker = build().markers.find(item => item.shape === 'sell');
  assert.equal(marker.y, marker.priceY - 12);
  assert.equal(srNearestTradeMarker([marker], marker.x, marker.y), marker);
  assert.equal(srNearestTradeMarker([marker], marker.x + 30, marker.y), null);
});

test('buy and sell entries remain on opposite sides of their price', () => {
  const buy = build({ direction: 'BUY' }).markers[0];
  const sell = build().markers[0];
  assert.equal(buy.y, buy.priceY + 12);
  assert.equal(sell.y, sell.priceY - 12);
});

test('exit tooltip uses account currency and identifies aggregated prices', () => {
  const text = build().markers.find(item => item.shape === 'exit').text;
  assert.match(text, /EUR/);
  assert.doesNotMatch(text, /USD/);
  assert.match(text, /平仓均价/);
  assert.match(text, /仓位 #42/);
  assert.match(text, /含手续费\/库存费/);
});

test('partial exits are not presented as a fully closed position', () => {
  const result = build({ closed: false, partial: true, close_volume: 0.05 });
  assert.equal(result.markers.length, 1);
  assert.equal(result.markers[0].shape, 'sell');
});

test('unknown or break-even profits are not coloured as a win', () => {
  for (const profit of [null, 0]) {
    const marker = build({ profit }).markers.find(item => item.shape === 'exit');
    assert.equal(marker.color, '#9aa4b2');
  }
  assert.equal(build({ profit: -2 }).markers.find(item => item.shape === 'exit').color, '#ef5350');
});

test('changing the timeframe picker does not reinterpret loaded candles', () => {
  const from = html.indexOf('function srChartBarSec()');
  const to = html.indexOf('function srChartAll()', from);
  const scope = vm.createContext({ srChart: { loadedTimeframe: 'H1' }, $: () => ({ value: 'M30' }) });
  vm.runInContext(html.slice(from, to), scope);
  assert.equal(scope.srChartBarSec(), 3600);
  scope.srChart.loadedTimeframe = 'M30';
  assert.equal(scope.srChartBarSec(), 1800);
});

test('hidden history produces neither markers nor connection lines', () => {
  const result = srTradeMarkers(candles, [], layout, 60, 'EUR');
  assert.equal(result.markers.length, 0);
  assert.equal(result.links.length, 0);
});
