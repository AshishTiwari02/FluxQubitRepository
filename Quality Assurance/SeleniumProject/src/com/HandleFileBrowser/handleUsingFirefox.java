package com.HandleFileBrowser;

import org.openqa.selenium.WebDriver;
//import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.firefox.FirefoxDriver;
import org.openqa.selenium.firefox.FirefoxOptions;
import org.openqa.selenium.firefox.FirefoxProfile;

public class handleUsingFirefox {

	public static void main(String[] args) throws InterruptedException {
		
		String driverPath = "webdriver.gecko.driver";
	    String driverURL = "D:\\Softwares\\Selenium\\geckodriver-v0.26.0-win64\\geckodriver.exe"; 
	    
	    System.setProperty(driverPath,driverURL);
	    
	    //Creating browser Profile for by passing the download popups and Downlaod window
	    FirefoxProfile profile = new FirefoxProfile();
	    profile.setPreference("browser.helperApps.neverask","text/plain");
	    profile.setPreference("browser.download.manager.showWhenStarting", false);
	    
	    //Create object for changing settings
	    FirefoxOptions option = new FirefoxOptions();
	    option.setProfile(profile);
	    
	    //Now create driver with add - on settings
	    WebDriver driver  = new FirefoxDriver(option);
	    
	    
	    String baseUrlFireFox = "https://sipcalculator.in/";
	    driver.get(baseUrlFireFox);
	    Thread.sleep(3000);
	    driver.close();
	    driver.quit();
	    

	}

}
