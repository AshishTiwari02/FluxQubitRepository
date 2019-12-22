package com.SikuliForDesktopApp;

import java.util.HashMap;
import java.util.concurrent.TimeUnit;

import org.openqa.selenium.By;
import org.openqa.selenium.Platform;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import org.openqa.selenium.remote.CapabilityType;
import org.openqa.selenium.remote.DesiredCapabilities;
import org.openqa.selenium.remote.RemoteWebDriver;
import org.sikuli.script.FindFailed;
import org.sikuli.script.Pattern;
import org.sikuli.script.Screen;


public class handleFile {

		static WebDriver driver;
	public static void main(String[] args) throws InterruptedException, FindFailed {
		
		// Change Chorme pref
		HashMap <String, Object> cpref =new HashMap<String, Object>();
		
		cpref.put("profile.default_content_settings.popups" , 0);
		cpref.put("download.prompt_for_download" ,false);
		
		ChromeOptions opt = new ChromeOptions();
		opt.setExperimentalOption("prefs", cpref);
		
		DesiredCapabilities cap = new DesiredCapabilities();
		cap.setCapability(CapabilityType.ACCEPT_SSL_CERTS , true);
		cap.setCapability(ChromeOptions.CAPABILITY, opt);
		
		
		/*
		 * DesiredCapabilities caps = new DesiredCapabilities() ;
		 * caps.setBrowserName("chrome"); cap.setPlatform(Platform.WIN10);
		 * 
		 * driver = new RemoteWebDriver(new url);
		 */
		
		String driverPath = "webdriver.chrome.driver";
		String driverURL = "D:\\Softwares\\Selenium\\chromedriver_win32\\chromedriver.exe";
		
		System.setProperty(driverPath, driverURL);
		WebDriver driver = new ChromeDriver();
		
		try {
		String urlRegis = "http://demo.automationtesting.in/Register.html";
		String imgFilePath = "C:\\Users\\Ashish\\eclipse-workspace\\SeleniumProject\\img\\";
		String fileInputPath = "D:\\Resume Folders\\Quality Assurance\\";
		
		driver.get(urlRegis);
		
		driver.manage().window().maximize();
		driver.manage().timeouts().implicitlyWait(3, TimeUnit.SECONDS);
		
		driver.findElement(By.xpath(".//*[@id=\"imagesrc\"]")).click();
		Screen scn = new Screen();
		
		Pattern fileTextBox = new Pattern(imgFilePath+"filepath.png");
		Pattern openButton = new Pattern(imgFilePath+"open.png");
		
		Thread.sleep(3000);
		
		scn.wait(fileTextBox,20);
		scn.type(fileTextBox, fileInputPath+"Cover_AshishTiwari.pdf");
		scn.click(openButton);		
		}
		catch (Exception e) 
		{
			System.out.println(e);
		}
		finally {
			driver.close();
			driver.quit();
			
		}
		
	}

}
