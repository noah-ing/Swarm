const { chromium } = require('playwright');

const task = process.argv[2] || 'Add input validation to the brain.py calculate_risk_score function';

(async () => {
  console.log('launching chromium...');
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });

  await page.goto('http://localhost:8420');
  await page.waitForTimeout(1000);
  console.log('dashboard loaded');

  // Clear the dialogue panel
  await page.evaluate(() => {
    document.getElementById('dialogue').innerHTML = '<div class="empty">waiting for dialogue...</div>';
  });

  await page.fill('#taskInput', task);
  console.log('task: ' + task);

  await page.click('#runBtn');
  console.log('running... watch the DIALOGUE panel');

  // Wait for completion
  await page.waitForSelector('#runBtn:not([disabled])', { timeout: 300000 });
  console.log('done!');

  // Keep browser open to view results
  await page.waitForTimeout(120000);
  await browser.close();
})();
