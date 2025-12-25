const { chromium } = require('playwright');

(async () => {
  console.log('launching chromium...');
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });

  await page.goto('http://localhost:8420');
  await page.waitForTimeout(2000);
  console.log('dashboard loaded');

  // Complex task that will trigger multi-agent dialogue
  // This targets brain.py which triggers HIGH/CRITICAL risk -> 2-3 proposers in dialogue
  const task = 'Refactor the calculate_risk_score function in brain.py to support weighted categories and add a normalize parameter';

  await page.fill('#taskInput', task);
  console.log('task entered: ' + task);

  await page.click('#runBtn');
  console.log('running task with multi-agent dialogue...');
  console.log('watch the DIALOGUE panel for agents talking to each other:');
  console.log('  - Proposer-Haiku, Proposer-Sonnet pitch solutions');
  console.log('  - Critic responds to their proposals');
  console.log('  - Proposers address critiques');
  console.log('  - Moderator synthesizes consensus');

  // wait for completion - dialogue involves multiple rounds of back-and-forth
  await page.waitForSelector('#runBtn:not([disabled])', { timeout: 600000 });
  await page.waitForTimeout(3000);
  console.log('task complete!');

  // show messages view to see full dialogue
  await page.click('.tab[data-view="messages"]');
  await page.waitForTimeout(2000);

  console.log('showing results for 60 seconds...');
  await page.waitForTimeout(60000);

  await browser.close();
  console.log('done');
})();
