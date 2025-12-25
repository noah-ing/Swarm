const { chromium } = require('playwright');

(async () => {
  console.log('launching chromium...');
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });

  await page.goto('http://localhost:8420');
  await page.waitForTimeout(2000);
  console.log('dashboard loaded');

  // high-risk task to trigger negotiation
  const task = 'Add a new function to brain.py called calculate_risk_score that takes a list of metrics and returns a weighted risk score';
  await page.fill('#taskInput', task);
  console.log('task entered');

  await page.click('#runBtn');
  console.log('running high-risk task with negotiation (this takes ~60-90 seconds)...');

  // wait for completion - negotiation takes time (multiple LLM calls)
  await page.waitForSelector('#runBtn:not([disabled])', { timeout: 300000 });
  await page.waitForTimeout(2000);
  console.log('task complete');

  // check events
  await page.screenshot({ path: '/tmp/swarm_negotiation_result.png' });

  // check messages tab
  await page.click('.tab[data-view="messages"]');
  await page.waitForTimeout(1000);
  await page.screenshot({ path: '/tmp/swarm_messages.png' });
  console.log('screenshots captured');

  // keep open to view
  console.log('keeping browser open for 30 seconds...');
  await page.waitForTimeout(30000);
  await browser.close();
  console.log('done');
})();
